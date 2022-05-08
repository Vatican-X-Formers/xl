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
import torch.optim as optim

from torch.nn.parallel import DistributedDataParallel
from train import parse_args, gen_model_config, sample_generation, \
            evaluate, get_boundary_config, train, weights_init, zero_init
from eval import load_checkpoint

import neptune.new as neptune
import utils
from data_utils import get_lm_corpus
from hourglass import MemTransformerLM
from utils.exp_utils import TimeoutHandler
from utils.exp_utils import create_exp_dir
from utils.exp_utils import l2_promote

import pdb
np.set_printoptions(suppress=True)


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
    boundary_kwargs['boundaries_type'] = 'noboundaries'

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

    # Finetune overwrite
    model_config['funnel_config'] = '(2, (8,), 2)'
    model_config['bp_mode'] = 'default'
    model_config['bp_capacity'] = 'nonlinear'
    model_config['bp_weight'] = None
    model_config['bp_switch_step'] = 500
    model_config['bp_target'] = ['entropy']
    model_config['mask_mode'] = 'boundary_ends_group'
    model_config['spikes_step'] = None
    model_config['spikes_left'] = 1
    model_config['spikes_right'] = 1
    model_config['value_perc'] = 100

    model = MemTransformerLM(**model_config)
    args.is_bp = True
    model.apply(functools.partial(weights_init, args=args))
    model.word_emb.apply(functools.partial(weights_init, args=args))

    checkpoint = load_checkpoint(args.ckpt_path)
    if list(checkpoint['model_state'].keys())[0].startswith('module'):
        model_state = {
            k[7:]: v for k, v in checkpoint['model_state'].items()
        }
    else:
        model_state = checkpoint['model_state']

    new_ms = {}

    for k, v in model_state.items():
        tmp = k.split('.')
        if tmp[0] == 'layers':
            a, b = (tmp[1]), int(tmp[2])
            if b < 2:
                a, b = 0, b
            elif b < 10:
                a, b = 2, b - 2
            else:
                a, b = 4, b - 10
            tmp[1] = str(a)
            tmp[2] = str(b)
            new_ms['.'.join(tmp)] = v
        else:
            new_ms[k] = v

    model.load_state_dict(new_ms, strict=False)
    # model.load_state_dict(model_state)

    if args.bp_zero_init:
        print('zero ini')
        model.boundary_predictor.apply(zero_init)

    args.n_all_param = sum([p.nelement() for p in model.parameters()])
    args.is_bp = getattr(model, 'boundary_predictor', None) is not None

    if args.optim.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=args.lr,
                               betas=(args.adam_b1,
                                      args.adam_b2),
                               eps=args.adam_eps,
                               weight_decay=args.weight_decay)

    args.max_step = 10000
    args.warmup_step = 0
    args.batch_chunk = 2

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.max_step - args.warmup_step, eta_min=args.eta_min)

    model = model.to(device)

    evaluate(va_iters[0], model, args, step=0)
    import sys
    sys.exit()

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
        run = neptune.init('syzymon/hourglass-pytorch')
        run['model_config'] = model_config
        run['args'] = vars(args)
        run['branch'] = os.getenv('TRAX_BRANCH')
        run['exp_path'] = os.getenv('EXPERIMENT_PATH')
        run['slurm_jobid'] = os.getenv('SLURM_JOB_ID')
    else:
        run = False

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
                    args=args,
                    run=run
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
