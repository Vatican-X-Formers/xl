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

import argparse
import functools
import itertools
import logging
import math
import os
import shutil
import sys
import time
import warnings
from collections import defaultdict

import dllogger
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
try:
    from apex import amp
except ModuleNotFoundError:
    warnings.warn('APEX AMP is unavailable')
from torch.nn.parallel import DistributedDataParallel

import neptune.new as neptune
import utils
from data_utils import get_lm_corpus
from mem_transformer import MemTransformerLM
from train import parse_args, gen_model_config 
from utils.exp_utils import AverageMeter
from utils.exp_utils import TimeoutHandler
from utils.exp_utils import benchmark
from utils.exp_utils import create_exp_dir
from utils.exp_utils import l2_promote
from utils.exp_utils import log_env_info
from utils.exp_utils import register_ignoring_timeout_handler
import torch.nn.functional as F

import pdb

def load_checkpoint(path):
    print(f'Starting path: {path}')
    path = os.path.dirname(path)
    while path.split('/')[-1] in ['configs', 'xl', 'pytorch']:
        path = path = os.path.dirname(path)
 
    path = os.path.join(path, 'xl/pytorch/LM-TFM/checkpoint_best.pt')

    dst = f'cuda:{torch.cuda.current_device()}'
    print(f'Loading checkpoint from {path}')
    checkpoint = torch.load(path, map_location=dst)
    return checkpoint


def sample_generation(vocab, model, args, temp = 0.5, start_seq = [0], steps=100):
    # Turn on evaluation mode which disables dropout.
    if args.mem_len == 0:
        model.reset_length(tgt_len=args.eval_tgt_len,
                           ext_len=args.ext_len + args.tgt_len - args.eval_tgt_len,
                           mem_len=args.mem_len
                           )
    else:
        model.reset_length(tgt_len=args.eval_tgt_len,
                           ext_len=args.ext_len,
                           mem_len=args.mem_len + args.tgt_len - args.eval_tgt_len,
                           )

    start_len = len(start_seq)
    generated_sequence = start_seq

    with torch.no_grad():
        for i in range(steps):
            mems, target = None, None
            data = torch.tensor(generated_sequence).unsqueeze(1).cuda()

            enable_autocast = args.fp16 and args.amp == 'pytorch'
            with torch.cuda.amp.autocast(enable_autocast):
                logits = model(data, target, mems)
                probs = F.softmax(logits[-1, 0, :], dim = 0)
                next_index = probs.cpu().multinomial(num_samples=1, replacement=True).item()
                generated_sequence.append(next_index)

    model.reset_length(tgt_len=args.tgt_len,
                       ext_len=args.ext_len,
                       mem_len=args.mem_len
                       )

    return vocab.convert_to_sent(generated_sequence).replace(' ', '')


def main():
    args = parse_args()

    torch.cuda.set_device(args.local_rank)
    l2_promote()
    device = torch.device('cuda' if args.cuda else 'cpu')
    utils.distributed.init_distributed(args.cuda)

    if args.profile:
        try:
            pyprof.init(enable_function_stack=True)
        except NameError:
            warnings.warn('Called pyprof.init() but pyprof is not available')

    # Set the random seed manually for reproducibility.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ###########################################################################
    # Load data
    ###########################################################################
    corpus = get_lm_corpus(args.data, args.dataset, args.vocab)
    ntokens = len(corpus.vocab)
    vocab = corpus.vocab
    args.n_token = ntokens

    if args.mem_len == 0:
        eval_mem_len = 0
    else:
        eval_mem_len = args.mem_len + args.tgt_len - args.eval_tgt_len

    te_iter = corpus.get_iterator('test', args.eval_batch_size, args.eval_tgt_len, device=device,mem_len=eval_mem_len, ext_len=args.ext_len)

    data = [batch for batch in te_iter]
    batch = data[0]
    input_data, target, _, _ = batch
    
    ###########################################################################
    # Build the model
    ###########################################################################
    if not args.autoreg:
        checkpoint = load_checkpoint(args.ckpt_path)
        names = [x[0] for x in list(checkpoint['model_state'].items())]
        old_checkpoint = names[-1].startswith('crit')
    else:
        old_checkpoint = False

    model_config = gen_model_config(args, vocab, old_checkpoint=old_checkpoint)
    model = MemTransformerLM(**model_config)
    model = model.to(device)
    if len(model.layers) > 1:
        # Sometimes the non-parameter part of the model is not moved to the cuda
        model.layers[1].leftmost_group = model.layers[1].leftmost_group.to(device)

    ###########################################################################
    # Test
    ###########################################################################
    if args.autoreg:
        model.eval()
        with torch.no_grad():
            target_test_len = 100
            full_logits = model(input_data[:target_test_len, :1], None, None).cpu().detach()

            for i in range(target_test_len):
                print(i)
                last_logit = model(input_data[:i + 1, :1], None, None).cpu().detach()[-1]
                assert torch.allclose(last_logit, full_logits[i], atol=1e-6) 

        print('The model passed the autoregresivity test')
        return

    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    test_sample = 'mary was not permitted to see them or to speak in her own defence at the tribunal she refused to offer a written defence unless elizabeth would guarantee a verdict of not guilty which elizabeth would not do although the casket letters were accepted by the inquiry as genuine after a study of the handwriting and of the information contained therein and were generally held to be certain proof of guilt if authentic the inquiry reached the conclusion that nothing was proven from the start this could have been '
    test_sample = test_sample.replace(' ', '_') 
    
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

    with torch.no_grad():
        print(sample_generation(vocab, model, args, start_seq=vocab.get_indices(test_sample), temp=1.0))

    if not args.debug and not args.no_eval:
        # Run on test data.
        summary = {}
        test_start_time = time.time()
        with torch.autograd.profiler.emit_nvtx(enabled=args.profile):
            test_loss = evaluate(te_iter, model, args)
            test_loss = utils.distributed.all_reduce_item(test_loss, 'mean')
        test_elapsed = time.time() - test_start_time

        logging.info('=' * 100)
        if args.dataset in ['enwik8', 'text8']:
            logging.info('| End of training | test time: {:5.2f}s | test loss {:5.2f} | test bpc {:9.5f}'.format(
                test_elapsed, test_loss, test_loss / math.log(2)))
        else:
            logging.info('| End of training | test time: {:5.2f}s | test loss {:5.2f} | test ppl {:9.3f}'.format(
                test_elapsed, test_loss, math.exp(test_loss)))
        if run:
            run['test_loss'].log(test_loss, step=train_step)
            run['test_ppl'].log(math.exp(test_loss), step=train_step)

        logging.info('=' * 100)

        summary.update({
            'test_elapsed': test_elapsed,
            'test_loss': test_loss,
            })

        if args.dataset in ['enwik8', 'text8']:
            summary['test_bits_per_character'] = test_loss / math.log(2)
        else:
            summary['test_perplexity'] = math.exp(test_loss)

if __name__ == "__main__":
    main()
