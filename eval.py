# coding: utf-8

# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import math
import os
import time

import numpy as np
import torch

import utils
from data_utils import get_lm_corpus
from hourglass import MemTransformerLM
from train import parse_args, gen_model_config, sample_generation, evaluate
from utils.exp_utils import l2_promote

import pdb

def load_checkpoint(path):
    print(f'Starting path: {path}')
    path = os.path.dirname(path)
    while path.split('/')[-1] in ['configs', 'xl']:
        path = path = os.path.dirname(path)

    path = os.path.join(path, 'xl/LM-TFM/checkpoint_best.pt')

    dst = f'cuda:{torch.cuda.current_device()}'
    print(f'Loading checkpoint from {path}')
    checkpoint = torch.load(path, map_location=dst)
    return checkpoint


def main():
    args = parse_args()

    torch.cuda.set_device(args.local_rank)
    l2_promote()
    device = torch.device('cuda' if args.cuda else 'cpu')
    utils.distributed.init_distributed(args.cuda)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ###########################################################################
    # Load data
    ###########################################################################
    boundary_kwargs = {
        'move_prob': args.move_prob,
        'deletion_prob': args.deletion_prob,
        'insert_prob':args.insert_prob,
        'clamp_group_sizes':args.clamp_group_sizes,
        'min_group_length':args.min_group_length,
        'max_group_length':args.max_group_length,
        'mean_normal':args.mean_normal,
        'std_normal':args.std_normal,
        'boundary_ids':args.boundary_ids,
        'boundaries_type':args.boundaries_type,
        'boundaries_tokens':args.boundaries_tokens,
    }
    corpus = get_lm_corpus(args.data, 
                           args.dataset,
                           **boundary_kwargs)
    ntokens = len(corpus.vocab)
    vocab = corpus.vocab
    args.n_token = ntokens

    eval_tgt_lengths = [2048]
    # eval_tgt_lengths = args.eval_tgt_lengths
    eval_total_lengths = [2048]
    # eval_total_lengths = args.eval_total_lengths

    va_iters, te_iters = [], []

    for eval_tgt_len, eval_total_len in zip(eval_tgt_lengths, eval_total_lengths):
        if args.mem_len == 0:
            eval_ext_len = eval_total_len - eval_tgt_len
        else:
            assert args.ext_len == 0
            eval_ext_len = 0
            
        va_iter = corpus.get_iterator('valid', args.eval_batch_size,
                                      eval_tgt_len, device,
                                      eval_ext_len, **boundary_kwargs)
        te_iter = corpus.get_iterator('test', args.eval_batch_size,
                                      eval_tgt_len, device,
                                      eval_ext_len, **boundary_kwargs)
        va_iters.append(va_iter)
        te_iters.append(te_iter)

    print(args)

    print(f'I focus on the test iter with args tgt_len = {eval_tgt_lengths[0]} ext_len = {eval_total_lengths[0] - eval_tgt_lengths[0]}')
    te_iter = te_iters[0]

    world_size = utils.distributed.get_world_size()
    print(f'We expect {te_iter.n_batch} batches')

    data = [batch for batch in te_iter]
    batch = data[0]
    input_data, target, seq_len, boundaries = batch

    ###########################################################################
    # Build the model
    ###########################################################################
    model_config = gen_model_config(args, vocab)
    model = MemTransformerLM(**model_config)
    model = model.to(device)

    ###########################################################################
    # Test
    ###########################################################################
    if args.autoreg:
        model.eval()
        with torch.no_grad():
            pdb.set_trace()
            target_test_len = 100
            full_logits = model(input_data[:target_test_len, :1], None, None, boundaries[:target_test_len, :1]).cpu().detach()

            for i in range(target_test_len):
                print(i)
                last_logit = model(input_data[:i + 1, :1], None, None, boundaries[:i + 1, :1]).cpu().detach()[-1]
                assert torch.allclose(last_logit, full_logits[i], atol=1e-6) 

        print('The model passed the autoregresivity test')
        return

    checkpoint = load_checkpoint(args.ckpt_path)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    # Evaluation part, loss comparison for different boundary heuristics

    target_test_len = 1000

    boundary_extractor = BoundaryExtractor()

    with torch.no_grad():
        loss = model(input_data[:target_test_len, :1], target[:target_test_len, :1], None, boundaries[:target_test_len, :1])[0].cpu().detach()
        loss = loss[:, 0]

    pdb.set_trace()

    with torch.no_grad():
        single_losses = []
        data_to_eval = input_data[:target_test_len, 0].cpu().numpy()
        text = vocab.convert_to_sent(data_to_eval).replace(' ',
                                                           '').replace('_', ' ')
        current_len = 0

        for idx, word in enumerate(text.split(' ')):
            for i in range(len(word)):
                word_boundaries = boundary_extractor.get_boundaries(word[:i + 1])
                current_x = input_data[:current_len + i + 1, :1]
                current_y = target[:current_len + i + 1, :1]
                current_boundaries = torch.cat([boundaries[:current_len, :1],
                                                word_boundaries[:i + 1, :1]], dim=0)
                single_loss = model(current_x, current_y, None, current_boundaries)
                single_losses.append(single_loss[0].cpu().detach()[-1, -1])

            #boundaries = torch.cat([current_boundaries,
             #                       torch.ones((1, 1)).bool().cuda()], dim=0)
            current_len += len(word)

            if not (idx + 1 == len(text.split(' '))):
                current_len += 1 # space
                single_loss = model(input_data[:current_len, :1],
                                    target[:current_len, :1],
                                    None,
                                    boundaries[:current_len, :1])
                single_losses.append(single_loss[0].cpu().detach()[-1,
                                                                   -1].item())

        pdb.set_trace()
        assert current_len == target_test_len
        single_losses = torch.tensor(single_losses)

        pdb.set_trace()

    def test_text8():
        test_sample = 'mary was not permitted to see them or to speak in her own defence at the tribunal she refused to offer a written defence unless elizabeth would guarantee a verdict of not guilty which elizabeth would not do although the casket letters were accepted by the inquiry as genuine after a study of the handwriting and of the information contained therein and were generally held to be certain proof of guilt if authentic the inquiry reached the conclusion that nothing was proven from the start this could have been '
        test_sample = test_sample.replace(' ', '_') 
        print(f'The test sample is {test_sample}')
        print('The completion is:')

        with torch.no_grad():
            print(sample_generation(vocab, model, args, start_seq=vocab.get_indices(test_sample)))
        print()
        print('Now we generate from the beginning')
        for i in range(3):
            with torch.no_grad():
                print(sample_generation(vocab, model, args, start_seq=vocab.get_indices(test_sample), steps=500))

    def test1():
        # https://arxiv.org/pdf/1808.04444v2.pdf
        possibilities = "prz, proven, proved, proof, prevented, presented, problematic, probably, provided, practical, provoked, preceded, predicted, previously, presumed, praised, proposed, practicable, produced, present, preserved, precisely, prior, protected, probable, prompted, proofed, properly, practiced, prohibited, profound, preferable, proceeded, precise, predictable, practically"
        pos = possibilities.split(', ')

        tmp = []
    
        for word in pos:
            test_to_make = test_sample + word
            test_to_make = test_to_make.replace(' ', '_')
            start_seq = vocab.get_indices(test_to_make)
            start_seq = torch.tensor(start_seq).unsqueeze(1).cuda()
            logits = model(start_seq, None, None)
            logits = logits[len(test_sample) - 1:, 0, :]
            fn = torch.nn.functional.log_softmax
            probs = fn(logits, dim = -1)
            x = vocab.get_indices(word)
            suma = probs.cpu()[(torch.arange(len(x)), x)].sum().item()
            tmp.append((word, suma)) 

    def sample_get_to_text(x):
        y = [int(lel) if lel != '<eos>' else ord('\n') for lel in x.split(' ')]
        return bytes(y).decode()
    
    if not args.debug and not args.no_eval:
        # Run on test data.
        test_start_time = time.time()
        with torch.autograd.profiler.emit_nvtx(enabled=args.profile):
            test_loss = evaluate(te_iter, model, args)
            test_loss = utils.distributed.all_reduce_item(test_loss, 'mean')
        test_elapsed = time.time() - test_start_time

        if args.dataset in ['enwik8', 'text8']:
            print('| End of training | test time: {:5.2f}s | test loss {:5.2f} | test bpc {:9.5f}'.format(
                test_elapsed, test_loss, test_loss / math.log(2)))
        else:
            logging.info('| End of training | test time: {:5.2f}s | test loss {:5.2f} | test ppl {:9.3f}'.format(
                test_elapsed, test_loss, math.exp(test_loss)))


if __name__ == "__main__":
    main()
