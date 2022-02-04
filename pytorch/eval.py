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
from utils.exp_utils import AverageMeter
from utils.exp_utils import TimeoutHandler
from utils.exp_utils import benchmark
from utils.exp_utils import create_exp_dir
from utils.exp_utils import l2_promote
from utils.exp_utils import log_env_info
from utils.exp_utils import register_ignoring_timeout_handler
import torch.nn.functional as F

import pdb
run = None

def parse_args():
    parent_parser = argparse.ArgumentParser(
        description='PyTorch Transformer-XL Language Model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False,
        )

    parser = argparse.ArgumentParser(parents=[parent_parser], add_help=True)
    cfg_parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)

    cfg_parser.add_argument('--config', default='default')
    cfg_parser.add_argument('--config_file', default=None)
    config_args, _ = cfg_parser.parse_known_args()

    if config_args.config is not None and config_args.config_file is not None:
        with open(config_args.config_file) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)[config_args.config]['train']
    else:
        config = {}

    general = parser.add_argument_group('general setup')
    general.add_argument('--work_dir', default='LM-TFM', type=str,
                         help='Directory for the results')
    general.add_argument('--append_dataset', action='store_true',
                         help='Automatically append dataset name to work_dir')
    general.add_argument('--append_time', action='store_true',
                         help='Automatically append current time to work_dir')
    general.add_argument('--cuda', action='store_true',
                         help='Run training on a GPU using CUDA')
    general.add_argument('--fp16', action='store_true',
                         help='Run training in fp16/mixed precision')
    general.add_argument('--restart', type=str, default='',
                         help='Restart training from the saved checkpoint')
    general.add_argument('--debug', action='store_true',
                         help='Run in debug mode (do not create exp dir)')
    general.add_argument('--log_all_ranks', action='store_true',
                         help='Enable logging from all distributed ranks')
    general.add_argument('--dllog_file', type=str, default='train_log.json',
                         help='Name of the DLLogger output file')
    general.add_argument('--txtlog_file', type=str, default='train_log.log',
                         help='Name of the txt log file')
    general.add_argument('--save_all', action='store_true',
                         help='Save all checkpoints')
    general.add_argument('--no_env', action='store_true',
                         help='Do not print info on execution env')
    general.add_argument('--no_eval', action='store_true',
                         help='Disable model evaluation')
    general.add_argument('--log_interval', type=int, default=10,
                         help='Report interval')
    general.add_argument('--target_throughput', type=float, default=None,
                         help='Target training throughput (for benchmarking)')
    general.add_argument('--target_perplexity', type=float, default=None,
                         help='Target validation perplexity (for benchmarking)')
    general.add_argument('--apex_amp_opt_level', type=str, default='O2',
                         choices=['O0', 'O1', 'O2', 'O3'],
                         help='Optimization level for apex amp')
    general.add_argument('--amp', choices=['apex', 'pytorch'], default='apex',
                         help='Implementation of automatic mixed precision')
    general.add_argument('--affinity', type=str,
                         default='socket_unique_interleaved',
                         choices=['socket', 'single', 'single_unique',
                                  'socket_unique_interleaved',
                                  'socket_unique_continuous',
                                  'disabled'],
                         help='type of CPU affinity')
    general.add_argument('--profile', action='store_true',
                         help='Enable profiling with DLProf')

    dataset = parser.add_argument_group('dataset setup')
    dataset.add_argument('--data', type=str, default='../data/wikitext-103',
                         help='Location of the data corpus')
    dataset.add_argument('--dataset', type=str, default='wt103',
                         choices=['wt103', 'lm1b', 'enwik8', 'text8'],
                         help='Dataset name')
    dataset.add_argument('--vocab', type=str, default='word', choices=['word', 'bpe'],
                         help='Type of vocabulary')

    model = parser.add_argument_group('model setup')
    model.add_argument('--n_layer', type=int, default=16,
                       help='Number of total layers')
    model.add_argument('--n_head', type=int, default=8,
                       help='Number of heads')
    model.add_argument('--d_head', type=int, default=64,
                       help='Head dimension')
    model.add_argument('--d_embed', type=int, default=-1,
                       help='Embedding dimension')
    model.add_argument('--d_model', type=int, default=512,
                       help='Model dimension')
    model.add_argument('--d_inner', type=int, default=2048,
                       help='Inner dimension in feedforward layer')
    model.add_argument('--dropout', type=float, default=0.1,
                       help='Global dropout rate')
    model.add_argument('--dropatt', type=float, default=0.0,
                       help='Attention probability dropout rate')
    model.add_argument('--pre_lnorm', action='store_true',
                       help='Apply LayerNorm to the input instead of the output')
    model.add_argument('--attn_type', type=int, default=0,
                       help='Attention type. 0 for ours, 1 for Shaw et al,'
                       '2 for Vaswani et al, 3 for Al Rfou et al.')
    model.add_argument('--not_tied', action='store_true',
                       help='Do not tie the word embedding and softmax weights')
    model.add_argument('--clamp_len', type=int, default=-1,
                       help='Use the same pos embeddings after clamp_len') # XDDDD
    model.add_argument('--adaptive', action='store_true',
                       help='Use adaptive softmax')
    model.add_argument('--div_val', type=int, default=1,
                       help='Dividend value for adaptive input and softmax')
    model.add_argument('--sample_softmax', type=int, default=-1,
                       help='Number of samples in sampled softmax')
    model.add_argument('--init', default='normal', type=str,
                       help='Parameter initializer to use')
    model.add_argument('--emb_init', default='normal', type=str,
                       help='Parameter initializer to use')
    model.add_argument('--init_range', type=float, default=0.1,
                       help='Parameters initialized by U(-init_range, init_range)')
    model.add_argument('--emb_init_range', type=float, default=0.01,
                       help='Parameters initialized by U(-init_range, init_range)')
    model.add_argument('--init_std', type=float, default=0.02,
                       help='Parameters initialized by N(0, init_std)')
    model.add_argument('--proj_init_std', type=float, default=0.01,
                       help='Parameters initialized by N(0, init_std)')
    model.add_argument('--funnel_config', type=str, default="[12, (0, 1) ,0]",
        help="[pre_funnel_vanilla_layers, (funnel_layers, shorten_factor), post_funnel_vanilla_layers]")
    model.add_argument('--funnel_resample', type=str, default='naive', help='')
    model.add_argument('--activation_function', type=str, default='relu', help='')

    opt = parser.add_argument_group('optimizer setup')
    opt.add_argument('--optim', default='adam', type=str,
                     choices=['adam', 'sgd', 'adagrad', 'lamb', 'jitlamb'],
                     help='Optimizer to use')
    opt.add_argument('--lr', type=float, default=0.00025,
                     help='Initial learning rate')
    opt.add_argument('--mom', type=float, default=0.0,
                     help='Momentum for sgd')
    opt.add_argument('--scheduler', default='cosine', type=str,
                     choices=['cosine', 'inv_sqrt', 'dev_perf', 'constant'],
                     help='LR scheduler to use')
    opt.add_argument('--max_step_scheduler', type=int, default=None,
                     help='Max number of training steps for LR scheduler')
    opt.add_argument('--warmup_step', type=int, default=1000,
                     help='Number of iterations for LR warmup')
    opt.add_argument('--lr_min', type=float, default=0.0,
                     help='Minimum learning rate during annealing')
    opt.add_argument('--clip', type=float, default=0.25,
                     help='Gradient clipping')
    opt.add_argument('--weight_decay', type=float, default=0.0,
                     help='Weight decay for adam|lamb')
    opt.add_argument('--clip_nonemb', action='store_true',
                     help='Only clip the gradient of non-embedding params')
    opt.add_argument('--patience', type=int, default=0,
                     help='Patience')
    opt.add_argument('--eta_min', type=float, default=0.000,
                     help='Min learning rate for cosine scheduler')

    training = parser.add_argument_group('training setup')
    training.add_argument('--max_step', type=int, default=40000,
                          help='Max number of training steps')
    training.add_argument('--batch_size', type=int, default=256,
                          help='Global batch size')
    training.add_argument('--local_batch_size', type=int, default=None,
                          help='Local (per-device) batch size, this setting \
                          overrides global --batch_size and sets batch_size \
                          to local_batch_size * world_size')
    training.add_argument('--batch_chunk', type=int, default=1,
                          help='Split batch into chunks and train with '
                          'gradient accumulation')
    training.add_argument('--roll', action='store_true', # XXX
                          help='Enable random shifts within each data stream')
    training.add_argument('--tgt_len', type=int, default=192, # XXX
                          help='Number of tokens to predict')
    training.add_argument('--ext_len', type=int, default=0, # XXX
                          help='Length of the extended context')
    training.add_argument('--mem_len', type=int, default=192, # XXX
                          help='Length of the retained previous heads')
    training.add_argument('--seed', type=int, default=1111,
                          help='Random seed')
    training.add_argument('--multi_gpu', default=None, type=str,
                          choices=['ddp', 'dp'],
                          help='Use multiple GPU')
    training.add_argument('--gpu0_bsz', type=int, default=-1,
                          help='Batch size on gpu 0 (for "dp" backend)')
    training.add_argument('--same_length', action='store_true', # XXX
                          help='Use the same attn length for all tokens')
    training.add_argument('--swap_mem', action='store_true',
                          help='Swap memory tensors to cpu')

    val = parser.add_argument_group('validation setup')
    val.add_argument('--eval_tgt_len', type=int, default=192,
                     help='Number of tokens to predict for evaluation')
    val.add_argument('--eval_batch_size', type=int, default=16,
                     help='Eval batch size')
    val.add_argument('--eval_max_steps', type=int, default=-1,
                     help='Max eval steps')
    val.add_argument('--eval_interval', type=int, default=5000,
                     help='Evaluation interval')
    val.add_argument('--ckpt_path', type=str, default="",
                     help='xd')


    dist = parser.add_argument_group('distributed setup')
    dist.add_argument('--local_rank',  type=int,
                      default=os.getenv('LOCAL_RANK', 0),
                      help='Used for multi-process training.')

    parser.set_defaults(**config)
    args, _ = parser.parse_known_args()

    args.tied = not args.not_tied

    if args.d_embed < 0:
        args.d_embed = args.d_model

    if args.ext_len < 0:
        raise RuntimeError('Extended context length must be non-negative')

    if args.mem_len == 0:
        if args.eval_tgt_len > args.ext_len + args.tgt_len:
            raise RuntimeError('eval_tgt_len should be <= tgt_len + ext_len; '
                               f'eval_tgt_len: {args.eval_tgt_len}, '
                               f'tgt_len: {args.tgt_len}, '
                               f'ext_len: {args.ext_len}')
    else:
        if args.eval_tgt_len > args.mem_len + args.tgt_len:
            raise RuntimeError('eval_tgt_len should be <= tgt_len + mem_len; '
                               f'eval_tgt_len: {args.eval_tgt_len}, '
                               f'tgt_len: {args.tgt_len}, '
                               f'mem_len: {args.mem_len}')

    if args.batch_size % args.batch_chunk != 0:
        raise RuntimeError('Batch size needs to be divisible by batch chunk')

    if (
        args.local_batch_size is not None
        and args.local_batch_size % args.batch_chunk != 0
    ):
        raise RuntimeError('Local batch size needs to be divisible by '
                           'batch chunk')

    if args.fp16 and args.amp == 'apex' and 'apex' not in sys.modules:
        raise RuntimeError(
            'APEX AMP unavailable, install APEX or switch to pytorch AMP'
        )

    return args


def load_checkpoint(path):
    if os.path.isdir(path):
        path = os.path.join(path, 'checkpoint_last.pt')

    dst = f'cuda:{torch.cuda.current_device()}'
    logging.info(f'Loading checkpoint from {path}')
    checkpoint = torch.load(path, map_location=dst)
    return checkpoint


def sample_generation(vocab, model, args, temp = 0.5, start_seq = [0], steps=100):
    # Turn on evaluation mode which disables dropout.
    model.eval()

    # If the model does not use memory at all, make the ext_len longer.
    # Otherwise, make the mem_len longer and keep the ext_len the same.
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

    generated_sequence = start_seq

    with torch.no_grad():
        for i in range(steps):
            mems, target = None, None
            data = torch.tensor(generated_sequence).unsqueeze(1).cuda()

            enable_autocast = args.fp16 and args.amp == 'pytorch'
            with torch.cuda.amp.autocast(enable_autocast):
                logits = model(data, target, mems)
                # pdb.set_trace()
                probs = F.softmax(logits[-1, 0, :], dim = 0)
                next_index = probs.cpu().multinomial(num_samples=1, replacement=True).item()
                generated_sequence.append(next_index)

    model.reset_length(tgt_len=args.tgt_len,
                       ext_len=args.ext_len,
                       mem_len=args.mem_len
                       )
    model.train()
    return vocab.convert_to_sent(generated_sequence)


def evaluate(eval_iter, model, args):
    # Turn on evaluation mode which disables dropout.
    model.eval()

    # If the model does not use memory at all, make the ext_len longer.
    # Otherwise, make the mem_len longer and keep the ext_len the same.
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

    # Evaluation
    total_len, total_loss = 0, 0.
    with torch.no_grad():
        mems = None
        for i, (data, target, seq_len, warm) in enumerate(eval_iter):
            if args.eval_max_steps > 0 and i >= args.eval_max_steps:
                break
            enable_autocast = args.fp16 and args.amp == 'pytorch'
            with torch.cuda.amp.autocast(enable_autocast):
                loss, mems, _ = model(data, target, mems)
                loss = loss.float().mean().type_as(loss)
            if warm:
                # assert (mems is None) or mems.size(1) == model.mem_len
                total_loss += seq_len * loss.item()
                total_len += seq_len

    # Switch back to the training mode
    model.reset_length(tgt_len=args.tgt_len,
                       ext_len=args.ext_len,
                       mem_len=args.mem_len
                       )
    model.train()

    return total_loss / total_len


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
    train_iter = corpus.get_iterator('train', args.eval_batch_size, args.eval_tgt_len, device=device,mem_len=eval_mem_len, ext_len=args.ext_len)
    data = [batch for batch in te_iter]
    batch = data[0]
    input_data, target, _, _ = batch

    # adaptive softmax / embedding
    cutoffs, tie_projs = [], [False]
    if args.adaptive:
        assert args.dataset in ['wt103', 'lm1b']
        if args.dataset == 'wt103':
            cutoffs = [19997, 39997, 199997]
            tie_projs += [True] * len(cutoffs)
        elif args.dataset == 'lm1b':
            cutoffs = [59997, 99997, 639997]
            tie_projs += [False] * len(cutoffs)

    ###########################################################################
    # Build the model
    ###########################################################################
    test_path = os.path.join(args.ckpt_path, 'checkpoint_best.pt')
    checkpoint = load_checkpoint(test_path)
    names = [x[0] for x in list(checkpoint['model_state'].items())]
    old_ckpt =  names[-1].startswith('crit')

    model_config = {
        'n_token': ntokens,
        'n_layer': args.n_layer,
        'n_head': args.n_head,
        'd_model': args.d_model,
        'd_head': args.d_head,
        'd_inner': args.d_inner,
        'dropout': args.dropout,
        'dropatt': args.dropatt,
        'dtype': None,
        'tie_weight': args.tied,
        'd_embed': args.d_embed,
        'div_val': args.div_val,
        'tie_projs': tie_projs,
        'pre_lnorm': args.pre_lnorm,
        'tgt_len': args.tgt_len,
        'ext_len': args.ext_len,
        'mem_len': args.mem_len,
        'cutoffs': cutoffs,
        'same_length': args.same_length,
        'attn_type': args.attn_type,
        'clamp_len': args.clamp_len,
        'sample_softmax': args.sample_softmax,
        'funnel_config': args.funnel_config,
        'funnel_resample': args.funnel_resample,
        'activation_function': args.activation_function,
        'old_checkpoint': old_ckpt
        }

    model = MemTransformerLM(**model_config)
    model = model.to(device)
    if len(model.layers) > 1:
        model.layers[1].leftmost_group = model.layers[1].leftmost_group.to(device)

    ###########################################################################
    # Test
    ###########################################################################
    model.load_state_dict(checkpoint['model_state'])
    fn = torch.nn.functional.log_softmax

    test_sample = 'mary was not permitted to see them or to speak in her own defence at the tribunal she refused to offer a written defence unless elizabeth would guarantee a verdict of not guilty which elizabeth would not do although the casket letters were accepted by the inquiry as genuine after a study of the handwriting and of the information contained therein and were generally held to be certain proof of guilt if authentic the inquiry reached the conclusion that nothing was proven from the start this could have been '
    test_converted = test_sample.replace(' ', '_') 
    possibilities = "prz, proven, proved, proof, prevented, presented, problematic, probably, provided, practical, provoked, preceded, predicted, previously, presumed, praised, proposed, practicable, produced, present, preserved, precisely, prior, protected, probable, prompted, proofed, properly, practiced, prohibited, profound, preferable, proceeded, precise, predictable, practically"
    pos = possibilities.split(', ')
    print(pos)

    def bytes_to_text(x):
        # assumes x to be on cuda, probably one dimensional tensor
        tmp = vocab.get_symbols(x.cpu().tolist())
        y = [int(lel) if lel != '<eos>' else ord('\n') for lel in tmp]
        return bytes(y).decode()

    def sample_get_to_text(x):
        y = [int(lel) if lel != '<eos>' else ord('\n') for lel in x.split(' ')]
        return bytes(y).decode()

    model.eval()

    with torch.no_grad():
        tmp = model(input_data[:500, :1], target[:500, :1], None)[0]
        logits = model(input_data[:500, :1], None, None)

    sample_generation(vocab, model, args, start_seq=input_data[:5, 0].cpu().tolist(), temp=0.99)
    pdb.set_trace()
    sample_generation(vocab, model, args, start_seq=input_data[:500, 0].cpu().tolist(), temp=0.99)

    tmp = []
    
    for word in pos:
        test_to_make = test_sample + word
        test_to_make = test_to_make.replace(' ', '_')
        start_seq = vocab.get_indices(test_to_make)
        start_seq = torch.tensor(start_seq).unsqueeze(0).cuda()
        logits = model(start_seq, None, None)
        logits = logits[0, len(test_sample):, :]
        probs = fn(logits, dim = -1)
        x = vocab.get_indices(word)
        suma = probs.cpu()[(torch.arange(len(x)), x)].sum().item()
        tmp.append((word, suma))

    completion = 'edicted as the only conclusion that would satisfy elizabeth the authenticity of '
    completion_converted = completion.replace(' ', '_')
    completion_sentence = vocab.get_indices(completion_converted)

    def funny_test():
        input_x = vocab.get_indices(test_converted) + completion_sentence
        target = input_x[1:]
        input_x = input_x[:-1]
        assert len(target) == len(input_x)
        input_x = torch.tensor(input_x).unsqueeze(0).cuda()
        target = torch.tensor(target).unsqueeze(0).cuda()
        output = model(input, target, None)[0][0]
        text = ''.join(vocab.get_symbols(target.cpu().numpy().tolist()[0]))
        tmpxd = output.cpu().detach().numpy().tolist()
        list(zip(tmpxd, text))

    results = []

    for i in range(1, 100):
        xd = model(input_data[:i, :1], target[:i, :1], None)[0].cpu().detach()[-1, 0]
        results.append(xd)

    sample_generation(vocab, model, args)
    return

    summary = {}

    if not args.debug and not args.no_eval:
        # Run on test data.
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
