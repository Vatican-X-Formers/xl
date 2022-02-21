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
from hourglass import MemTransformerLM
from train import parse_args, gen_model_config, sample_generation, evaluate 
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
    corpus = get_lm_corpus(args.data, 
                           args.dataset,
                           move_prob=args.move_prob,
                           deletion_prob=args.deletion_prob,
                           insert_prob=args.insert_prob,
                           clamp_group_sizes=args.clamp_group_sizes,
                           min_group_length=args.min_group_length,
                           max_group_length=args.max_group_length,
                           mean_normal=args.mean_normal,
                           std_normal=args.std_normal,
                           boundary_ids=args.boundary_ids,
                           boundaries_type=args.boundaries_type,
                           boundaries_tokens=args.boundaries_tokens)
    ntokens = len(corpus.vocab)
    vocab = corpus.vocab
    args.n_token = ntokens

    if args.mem_len == 0:
        eval_mem_len = 0
    else:
        eval_mem_len = args.mem_len + args.tgt_len - args.eval_tgt_len

    print(args)
    # There should be 5*1e6 tokens in test set in both enwik and text8
    te_iter = corpus.get_iterator('test', args.eval_batch_size, args.eval_tgt_len, device=device,mem_len=eval_mem_len, ext_len=args.ext_len)
    world_size = utils.distributed.get_world_size()
    print(f'We expect {te_iter.data.size(0)/args.eval_tgt_len} batches')
    data = [batch for batch in te_iter]
    batch = data[0]
    input_data, target, seq_len, boundaries = batch

    ###########################################################################
    # Build the model
    ###########################################################################
    model_config = gen_model_config(args, vocab)
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

    with torch.no_grad():
        target_test_len = 100
        loss = model(input_data[:target_test_len, :1], target[:target_test_len, :1], None, boundaries[:target_test_len, :1])[0].cpu().detach()
        tab_losses = []
        for i in range(target_test_len):
            x = input_data[:i + 1, 0].cpu().numpy()
            sent = vocab.convert_to_sent(x)
            x_boundaries = corpus.boundary_creator.get_boundaries(sent)
            x_boundaries = x_boundaries.cuda()
            elem_loss = model(input_data[:i + 1, :1], target[:i + 1, :1], None, x_boundaries.unsqueeze(1))
            tab_losses.append(elem_loss[0][-1, -1])
        pdb.set_trace()
    
    test_sample = 'mary was not permitted to see them or to speak in her own defence at the tribunal she refused to offer a written defence unless elizabeth would guarantee a verdict of not guilty which elizabeth would not do although the casket letters were accepted by the inquiry as genuine after a study of the handwriting and of the information contained therein and were generally held to be certain proof of guilt if authentic the inquiry reached the conclusion that nothing was proven from the start this could have been '
    test_sample = test_sample.replace(' ', '_') 
    
    def test_text8():
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
