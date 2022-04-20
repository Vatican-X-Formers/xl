import argparse
import functools
import itertools
import math
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from torch.nn.parallel import DistributedDataParallel

import neptune.new as neptune
import utils
from data_utils import get_lm_corpus
from hourglass import MemTransformerLM
from utils.exp_utils import TimeoutHandler
from utils.exp_utils import create_exp_dir
import torch.nn.functional as F
import pdb
from utils.exp_utils import l2_promote
run = None
np.set_printoptions(suppress=True)


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

    assert config_args.config is not None and config_args.config_file is not None
    with open(config_args.config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)[config_args.config]['train']

    # Main args
    general = parser.add_argument_group('general setup')
    general.add_argument('--work_dir', default='LM-TFM', type=str,
                         help='Directory for the results')
    general.add_argument('--cuda', action='store_true',
                         help='Run training on a GPU using CUDA')
    general.add_argument('--debug', action='store_true',
                         help='Run in debug mode (do not create exp dir)')
    general.add_argument('--log_interval', type=int, default=10,
                         help='Report interval')
    general.add_argument('--affinity', type=str,
                         default='socket_unique_interleaved',
                         help='type of CPU affinity')

    dataset = parser.add_argument_group('dataset setup')
    dataset.add_argument('--data', type=str, default='../data/wikitext-103',
                         help='Location of the data corpus')
    dataset.add_argument('--dataset', type=str, default='wt103',
                         choices=['wt103', 'lm1b', 'enwik8', 'text8'],
                         help='Dataset name')

    model = parser.add_argument_group('model setup')
    model.add_argument('--n_head', type=int, default=8,
                       help='Number of heads')
    model.add_argument('--d_head', type=int, default=64,
                       help='Head dimension')
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
    model.add_argument('--clamp_len', type=int, default=-1,
                       help='Use the same pos embeddings after clamp_len')
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
    model.add_argument('--funnel_config', type=str, default="[3, (8,) ,3]",
                       help="[pre_funnel_vanilla_layers, (funnel_layers, ), post_funnel_vanilla_layers]")
    model.add_argument('--downsample_mode', type=str, default='average', help='')
    model.add_argument('--upsample_mode', type=str, default='average', help='')
    model.add_argument('--activation_function', type=str, default='relu', help='')
    model.add_argument('--gather_stats', nargs="+", default=['shortened_length'])
    model.add_argument('--bp_mode', type=str, default='none')
    model.add_argument('--bp_capacity', type=str, default='none')
    model.add_argument('--bp_weight', type=float, default=1.0)
    model.add_argument('--bp_switch_step', type=int, default=0)
    model.add_argument('--bp_target', type=str, nargs='+')
    model.add_argument('--rl_loss_combine', type=str, default='none')
    model.add_argument('--mask_mode', type=str, default='boundary_starts_group')

    boundaries = parser.add_argument_group('boundary creator')
    boundaries.add_argument('--move_prob', type=float, default=0.0)
    boundaries.add_argument('--deletion_prob', type=float, default=0.0)
    boundaries.add_argument('--insert_prob', type=float, default=0.0)
    boundaries.add_argument('--clamp_group_sizes', action='store_true')
    boundaries.add_argument('--min_group_length', type=int, default=0)
    boundaries.add_argument('--max_group_length', type=int, default=1000000)
    boundaries.add_argument('--mean_normal', type=float, default=5.5)
    boundaries.add_argument('--std_normal', type=float, default=1.0)
    boundaries.add_argument('--boundary_ids', type=str, default='[]')
    boundaries.add_argument('--boundaries_type', type=str, default='vanilla')
    boundaries.add_argument('--tokenizer_type', type=str)
    boundaries.add_argument('--tokenizer_vocab_size', type=int)
    boundaries.add_argument('--tokenizer_dropout', type=float)
    boundaries.add_argument('--tokenizer_save_dir', default='./tokenizer_data/')
    boundaries.add_argument('--tokenizer_algorithm', default=None)

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
    opt.add_argument('--clip', type=float, default=0.25,
                     help='Gradient clipping')
    opt.add_argument('--weight_decay', type=float, default=0.0,
                     help='Weight decay for adam')
    opt.add_argument('--adam_b1', type=float, default=0.9)
    opt.add_argument('--adam_b2', type=float, default=0.999)
    opt.add_argument('--adam_eps', type=float, default=1e-8)
    opt.add_argument('--eta_min', type=float, default=0.000,
                     help='Min learning rate for cosine scheduler')

    training = parser.add_argument_group('training setup')
    training.add_argument('--max_step', type=int, default=40000,
                          help='Max number of training steps')
    training.add_argument('--batch_size', type=int, default=256,
                          help='Global batch size')
    training.add_argument('--batch_chunk', type=int, default=1,
                          help='Split batch into chunks and train with '
                          'gradient accumulation')
    training.add_argument('--roll', action='store_true',
                          help='Enable random shifts within each data stream')
    training.add_argument('--shuffle', action='store_true')
    training.add_argument('--tgt_len', type=int, default=192,
                          help='Number of tokens to predict')
    training.add_argument('--ext_len', type=int, default=0,
                          help='Length of the extended context')
    training.add_argument('--seed', type=int, default=1111,
                          help='Random seed')
    training.add_argument('--multi_gpu', default=None, type=str,
                          choices=['ddp'],
                          help='Use multiple GPU')
    training.add_argument('--same_length', action='store_true',
                          help='Use the same attn length for all tokens')

    val = parser.add_argument_group('validation setup')
    val.add_argument('--eval_tgt_lengths', nargs="+")
    val.add_argument('--eval_total_lengths', nargs="+")
    val.add_argument('--eval_batch_size', type=int, default=16,
                     help='Eval batch size')
    val.add_argument('--eval_max_steps', type=int, default=-1,
                     help='Max eval steps')
    val.add_argument('--eval_interval', type=int, default=5000,
                     help='Evaluation interval')
    val.add_argument('--text_generation_interval', type=int, default=5000)
    val.add_argument('--ckpt_path', type=str, default="")
    val.add_argument('--autoreg', action='store_true')

    dist = parser.add_argument_group('distributed setup')
    dist.add_argument('--local_rank', type=int,
                      default=os.getenv('LOCAL_RANK', 0),
                      help='Used for multi-process training.')

    parser.set_defaults(**config)
    args, _ = parser.parse_known_args()

    args.ckpt_path = '/'.join(config_args.config_file.split('/')[:-1])

    assert len(args.eval_tgt_lengths) == len(args.eval_total_lengths)

    if args.batch_size % args.batch_chunk != 0:
        raise RuntimeError('Batch size needs to be divisible by batch chunk')

    return args


def save_checkpoint(args, model, model_config, optimizer, scheduler,
                    vocab, epoch, batch, last_iter, train_step, work_dir):
    state = {
        'args': args,
        'model_config': model_config,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'vocab': vocab,
        'epoch': epoch,
        'batch': batch,
        'last_iter': last_iter,
        'train_step': train_step,
        }

    last_chkpt_fname = 'checkpoint_last.pt'

    with utils.distributed.sync_workers() as rank:
        last_chkpt_path = os.path.join(work_dir, last_chkpt_fname)
        if rank == 0:
            print(f'Saving checkpoint to {last_chkpt_path}')
            torch.save(state, last_chkpt_path)


def load_checkpoint(path):
    if os.path.isdir(path):
        path = os.path.join(path, 'checkpoint_last.pt')

    dst = f'cuda:{torch.cuda.current_device()}'
    print(f'Loading checkpoint from {path}')
    checkpoint = torch.load(path, map_location=dst)
    return checkpoint


def init_weight(weight, args):
    if args.init == 'uniform':
        nn.init.uniform_(weight, -args.init_range, args.init_range)
    elif args.init == 'normal':
        nn.init.normal_(weight, 0.0, args.init_std)


def init_bias(bias):
    nn.init.constant_(bias, 0.0)


def weights_init(m, args):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            init_weight(m.weight, args)
        if hasattr(m, 'bias') and m.bias is not None:
            init_bias(m.bias)
    elif classname.find('Downsampler') != -1:
        if hasattr(m, 'leftmost_group'):
            init_weight(m.leftmost_group, args)
    elif classname.find('Embedding') != -1:
        if hasattr(m, 'weight'):
            init_weight(m.weight, args)
    elif classname.find('LayerNorm') != -1:
        if hasattr(m, 'weight'):
            nn.init.normal_(m.weight, 1.0, args.init_std)
        if hasattr(m, 'bias') and m.bias is not None:
            init_bias(m.bias)
    elif classname.find('TransformerLM') != -1:
        if hasattr(m, 'r_w_bias'):
            init_weight(m.r_w_bias, args)
        if hasattr(m, 'r_r_bias'):
            init_weight(m.r_r_bias, args)


def sample_generation(vocab, boundary_creator, model,
                      temp=1.0, start_seq=[0], steps=100,
                      dataset='text8', step=0):
    model.eval()

    start_len = len(start_seq)
    generated_sequence = start_seq

    with torch.no_grad():
        for i in range(steps):
            # Data preparation
            data = torch.tensor(generated_sequence).unsqueeze(1).cuda()
            boundaries = boundary_creator.get_boundaries(
                txt=vocab.convert_to_sent(data, mode='real'),
                tensor=data
            ).t().bool().contiguous()[:-1, :]

            # Forward through the model
            logits = model(
                data=data,
                target=None,
                boundaries=boundaries,
                step=step
            )

            # Transform logits to probs
            probs = F.softmax(logits[-1, 0, :], dim=0)

            # Sample next character
            next_index = probs.cpu().multinomial(num_samples=1, replacement=True).item()
            generated_sequence.append(next_index)

    model.train()

    generated_sample = vocab.convert_to_sent(generated_sequence,
                                             mode='real')

    return generated_sample


def evaluate(eval_iter, model, args, step):
    # Turn on evaluation mode which disables dropout.
    model.eval()

    stats_agg = defaultdict(list)
    # Evaluation
    total_len, total_loss = 0, 0.
    with torch.no_grad():
        for i, (data, target, seq_len, boundaries) in enumerate(eval_iter.get_fixlen_iter()):
            data = data.to(eval_iter.device, non_blocking=True)
            target = target.to(eval_iter.device, non_blocking=True)
            boundaries = boundaries.to(eval_iter.device, non_blocking=True)
            if args.eval_max_steps > 0 and i >= args.eval_max_steps:
                break
            loss, stats, aux_loss = model(data, target, boundaries, step=step)
            loss = loss.float().mean().type_as(loss)

            total_loss += seq_len * loss.item()
            total_len += seq_len

            for k, v in stats.items():
                stats_agg[k].append(v)

    model.train()

    return total_loss / total_len, stats_agg


def gen_model_config(args, vocab):
    ntokens = len(vocab)
    model_config = {
        'n_token': ntokens,
        'n_head': args.n_head,
        'd_model': args.d_model,
        'd_head': args.d_head,
        'd_inner': args.d_inner,
        'dropout': args.dropout,
        'dropatt': args.dropatt,
        'pre_lnorm': args.pre_lnorm,
        'same_length': args.same_length,
        'clamp_len': args.clamp_len,
        'funnel_config': args.funnel_config,
        'downsample_mode': args.downsample_mode,
        'upsample_mode': args.upsample_mode,
        'activation_function': args.activation_function,
        'gather_stats': args.gather_stats,
        'bp_mode': args.bp_mode,
        'bp_capacity': args.bp_capacity,
        'bp_weight': args.bp_weight,
        'bp_switch_step': args.bp_switch_step,
        'bp_target': args.bp_target,
        'rl_loss_combine': args.rl_loss_combine,
        'mask_mode': args.mask_mode,
        }

    return model_config


def get_boundary_config(args):
    boundary_config = {
        'move_prob': args.move_prob,
        'deletion_prob': args.deletion_prob,
        'insert_prob': args.insert_prob,
        'clamp_group_sizes': args.clamp_group_sizes,
        'min_group_length': args.min_group_length,
        'max_group_length': args.max_group_length,
        'mean_normal': args.mean_normal,
        'std_normal': args.std_normal,
        'boundary_ids': args.boundary_ids,
        'boundaries_type': args.boundaries_type,
        'tokenizer_type': args.tokenizer_type,
        'tokenizer_vocab_size': args.tokenizer_vocab_size,
        'tokenizer_dropout': args.tokenizer_dropout,
        'tokenizer_save_dir': args.tokenizer_save_dir,
        'tokenizer_algorithm': args.tokenizer_algorithm,
    }
    return boundary_config


def train_iteration(model, i, data_chunks, target_chunks, boundaries_chunks,
                    args, step):
    data_i = data_chunks[i].contiguous()
    target_i = target_chunks[i].contiguous()

    if boundaries_chunks is not None:
        boundaries_i = boundaries_chunks[i].contiguous()
    else:
        boundaries_i = None

    seq_loss, stats, aux_loss = model(data_i, target_i,
                                      boundaries=boundaries_i, step=step)
    seq_loss = seq_loss.float().mean().type_as(seq_loss)
    total_loss = (seq_loss + aux_loss) / args.batch_chunk

    total_loss.backward()

    return seq_loss.item() / args.batch_chunk, stats


def train(tr_iter, va_iters, model, model_config, optimizer,
          scheduler, vocab, epoch, last_iter, train_step,
          timeout_handler, args):
    # Turn on training mode which enables dropout.
    model.train()

    # Accumulates loss
    train_loss = 0

    # For measuring throuhput, accumulates # of tokens processed
    target_tokens = 0

    # Sometimes, in between epochs, I don't have even number of steps. Like
    # epoch is 60 steps and I want to log every 50. I log once in the first
    # epoch but in the second epoch I start from 60 and I'd like to log at 100
    # and not 110. Here the approach is to gather data from 40 steps, not move
    # it from previous epoch and divide by 40.
    log_step = 0

    # Values that I get in each step and average them out
    # There is a bug actually here, I gather the data only from 1 GPU
    # If that's data from training I should account for that
    stats_agg = defaultdict(list)

    log_start_time = time.time()
    train_iter = tr_iter.get_fixlen_iter(start=last_iter, shuffle=args.shuffle)

    for batch, (data, target, seq_len, boundaries) in enumerate(train_iter, start=1):
        data = data.to(tr_iter.device, non_blocking=True)
        target = target.to(tr_iter.device, non_blocking=True)
        if boundaries is not None:
            boundaries = boundaries.to(tr_iter.device, non_blocking=True)

        log_step += 1
        target_tokens += target.numel()

        for param in model.parameters():
            param.grad = None

        data_chunks = torch.chunk(data, args.batch_chunk, 1)
        target_chunks = torch.chunk(target, args.batch_chunk, 1)
        if boundaries is not None:
            boundaries_chunks = torch.chunk(boundaries, args.batch_chunk, 1)
        else:
            boundaries_chunks = None

        for i in range(args.batch_chunk):
            if i < args.batch_chunk - 1 and isinstance(model, DistributedDataParallel):
                with model.no_sync():
                    train_loss_chunk, stats = train_iteration(
                        model, i, data_chunks, target_chunks,
                        boundaries_chunks, args, train_step
                    )
            else:
                train_loss_chunk, stats = train_iteration(
                    model, i, data_chunks, target_chunks, boundaries_chunks,
                    args, train_step
                )

            train_loss += train_loss_chunk

        # Custom stats added by me
        for k, v in stats.items():
            stats_agg[k].append(v)

        stats_agg['grad_l2'].append(sum(p.grad.detach().data.norm(2).item() ** 2 for p in model.parameters()) ** 0.5)
        stats_agg['weights_l2'].append(sum(p.detach().norm(2).item() ** 2 for p in model.parameters()) ** 0.5)

        # if run and train_step % args.text_generation_interval == 0:
        #     generated_sample =sample_generation(vocab, model, args)
        #     run['gen/text'].log(generated_sample)
        # Finish of custom statistics

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        # step-wise learning rate annealing
        train_step += 1
        if args.scheduler == 'cosine':
            # linear warmup stage
            if train_step < args.warmup_step:
                curr_lr = args.lr * train_step / args.warmup_step
                optimizer.param_groups[0]['lr'] = curr_lr
            else:
                scheduler.step(train_step - args.warmup_step)
        else:
            raise NotImplementedError

        if train_step % args.log_interval == 0 or train_step == 1:
            cur_loss = train_loss / log_step
            cur_loss = utils.distributed.all_reduce_item(cur_loss, op='mean')
            train_loss = 0

            elapsed = time.time() - log_start_time
            avg_elapsed = elapsed / log_step
            avg_elapsed = utils.distributed.all_reduce_item(avg_elapsed, op='max')
            log_start_time = time.time()
            log_step = 0

            lr = optimizer.param_groups[0]['lr']
            throughput = target_tokens / elapsed
            throughput = utils.distributed.all_reduce_item(throughput, op='sum')
            target_tokens = 0

            log_str = '| epoch {:3d} step {:>8d} | batches {:>6d} / {:d} | lr {:.3e} ' \
                '| ms/batch {:5.1f} | tok/s {:7.0f} | loss {:5.2f}'.format(
                    epoch,
                    train_step,
                    batch,
                    tr_iter.n_batch,
                    lr,
                    avg_elapsed * 1000,
                    throughput,
                    cur_loss,
                )
            print_once(log_str, args)

            if run:
                run['lr'].log(lr, step=train_step)
                run['train/loss'].log(cur_loss, step=train_step)
                run['tokens_per_sec'].log(throughput, step=train_step)
                for k, v in stats_agg.items():
                    run[k].log(np.array(v).mean(), step=train_step)
                stats_agg = defaultdict(list)

        do_periodic_eval = train_step % args.eval_interval == 0
        is_final_step = train_step == args.max_step
        interrupted = timeout_handler.interrupted

        if (do_periodic_eval or is_final_step or interrupted):
            eval_start_time = time.time()

            eval_tgt_lengths = args.eval_tgt_lengths
            eval_total_lengths = args.eval_total_lengths

            val_losses = []

            for i, (eval_tgt_len, eval_total_len) in enumerate(zip(eval_tgt_lengths, eval_total_lengths)):
                val_loss, stats_val = evaluate(va_iters[i], model, args,
                                               train_step)
                val_loss = utils.distributed.all_reduce_item(val_loss, op='mean')
                val_losses.append(val_loss)
                if run:
                    run[f'val/loss_tgt{eval_tgt_len}_total{eval_total_len}'].log(val_loss, step=train_step)
                    for k, v in stats_val.items():
                        run[f'val/{k}'].log(np.array(v).mean(), step=train_step)

            # we assume that first one is the main one
            val_loss = val_losses[0]

            if run:
                run['val/loss'].log(val_loss, step=train_step)

            print_once('-' * 100, args)
            log_str = '| Eval {:3d} at step {:>8d} | time: {:5.2f}s ' \
                      '| valid loss {:5.2f}'.format(
                          train_step // args.eval_interval,
                          train_step,
                          (time.time() - eval_start_time),
                          val_loss,
                          )
            print_once(log_str, args)
            print_once('-' * 100, args)

            last_iter = tr_iter.last_iter

            if not args.debug:
                save_checkpoint(args, model, model_config, optimizer, scheduler,
                                vocab, epoch, batch, last_iter,
                                train_step, args.work_dir)

            log_start_time += time.time() - eval_start_time

        if interrupted:
            print_once('Received SIGTERM, exiting', args)
            sys.exit(0)

        if is_final_step:
            break

    return train_step


def print_once(txt, args):
    if args.local_rank == 0:
        print(txt)


def main():
    args = parse_args()

    # Nv speed improvement by 15%, assigning particular CPU threads and link to particular GPUs
    # It requires 4 threads/cpu per gpu
    if args.affinity != 'disabled':
        nproc_per_node = torch.cuda.device_count()
        affinity = utils.gpu_affinity.set_affinity(
            args.local_rank,
            nproc_per_node,
            args.affinity
        )
        print(f'{args.local_rank}: thread affinity: {affinity}')

    # Initialize distributed backend
    torch.cuda.set_device(args.local_rank)
    l2_promote()
    device = torch.device('cuda' if args.cuda else 'cpu')
    utils.distributed.init_distributed(args.cuda)
    with utils.distributed.sync_workers() as rank:
        if rank == 0:
            create_exp_dir(args.work_dir,
                           scripts_to_save=['train.py', 'hourglass.py'],
                           debug=args.debug)

    if args.debug:
        print_once('This run is in DEBUG mode', args)

    print_once(f'world size: {utils.distributed.get_world_size()}', args)

    # Set the random seed manually for reproducibility.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ###########################################################################
    # Load data
    ###########################################################################
    boundary_kwargs = get_boundary_config(args)
    corpus = get_lm_corpus(args.data,
                           args.dataset,
                           **boundary_kwargs)
    vocab = corpus.vocab
    args.n_token = len(vocab)

    eval_tgt_lengths = args.eval_tgt_lengths
    eval_total_lengths = args.eval_total_lengths

    tr_iter = corpus.get_iterator(split='train',
                                  bsz=args.batch_size,
                                  tgt_len=args.tgt_len,
                                  device=device,
                                  ext_len=args.ext_len,
                                  **boundary_kwargs)

    va_iters, te_iters = [], []

    for eval_tgt_len, eval_total_len in zip(eval_tgt_lengths, eval_total_lengths):
        eval_ext_len = eval_total_len - eval_tgt_len

        va_iter = corpus.get_iterator(split='valid',
                                      bsz=args.eval_batch_size,
                                      tgt_len=eval_tgt_len,
                                      device=device,
                                      ext_len=eval_ext_len,
                                      **boundary_kwargs)
        te_iter = corpus.get_iterator(split='test',
                                      bsz=args.eval_batch_size,
                                      tgt_len=eval_tgt_len,
                                      device=device,
                                      ext_len=eval_ext_len,
                                      **boundary_kwargs)
        va_iters.append(va_iter)
        te_iters.append(te_iter)

    ###########################################################################
    # Build the model
    ###########################################################################
    model_config = gen_model_config(args, vocab)
    model = MemTransformerLM(**model_config)
    model.apply(functools.partial(weights_init, args=args))
    model.word_emb.apply(functools.partial(weights_init, args=args))
    args.n_all_param = sum([p.nelement() for p in model.parameters()])

    # optimizer
    if args.optim.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
                              momentum=args.mom)
    elif args.optim.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr,
                               betas=(args.adam_b1, args.adam_b2),
                               eps=args.adam_eps,
                               weight_decay=args.weight_decay)

    # scheduler
    if args.max_step_scheduler:
        max_step = args.max_step_scheduler
    else:
        max_step = args.max_step

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, max_step - args.warmup_step, eta_min=args.eta_min)

    model = model.to(device)

    if args.multi_gpu == 'ddp' and torch.distributed.is_initialized():
        model = DistributedDataParallel(model,
                                             device_ids=[args.local_rank],
                                             output_device=args.local_rank,
                                             broadcast_buffers=False,
                                             find_unused_parameters=True,
                                             )

    # Log training and model args
    if rank == 0:
        # Neptune
        global run
        run = neptune.init('syzymon/hourglass-pytorch')
        run['model_config'] = model_config
        run['args'] = vars(args)
        run['branch'] = os.getenv('TRAX_BRANCH')
        run['exp_path'] = os.getenv('EXPERIMENT_PATH')
        run['slurm_jobid'] = os.getenv('SLURM_JOB_ID')

    if rank == 0:
        print(model)
        print('=' * 100)
        for k, v in args.__dict__.items():
            print('    - {} : {}'.format(k, v))
        print('=' * 100)

    ###########################################################################
    # Train
    ###########################################################################
    with TimeoutHandler() as timeout_handler:
        # At any point you can hit Ctrl + C to break out of training early.
        try:
            train_step = 0
            for epoch in itertools.count(start=1):
                if args.roll:
                    tr_iter.roll(seed=args.seed + epoch)
                train_step = train(
                    tr_iter, va_iters, model, model_config,
                    optimizer, scheduler,
                    vocab, epoch,
                    last_iter=0,
                    train_step=train_step,
                    timeout_handler=timeout_handler,
                    args=args
                    )

                if train_step == args.max_step:
                    print('End of training')
                    break
        except KeyboardInterrupt:
            sys.exit()
            print('Exiting from training early')

    ###########################################################################
    # Test
    ###########################################################################
    if not args.debug:
        test_losses = []

        for i, (eval_tgt_len, eval_total_len) in enumerate(zip(eval_tgt_lengths, eval_total_lengths)):
            test_loss, stats_test = evaluate(te_iters[i], model, args,
                                             train_step)
            test_loss = utils.distributed.all_reduce_item(test_loss, op='mean')
            test_losses.append(test_loss)
            if run:
                run[f'test/loss_tgt{eval_tgt_len}_total{eval_total_len}'].log(test_loss, step=train_step)
                for k, v in stats_test.items():
                    run[f'test/{k}'].log(np.array(v).mean(), step=train_step)

        test_loss = test_losses[0]

        print_once('| End of training | test loss {:5.2f} | test bpc {:9.5f}'.format(
            test_loss, test_loss / math.log(2)), args)
        if run:
            run['test_loss'].log(test_loss, step=train_step)


if __name__ == "__main__":
    main()
