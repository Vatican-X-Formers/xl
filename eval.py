import os
import time

import numpy as np
import torch

import utils
from data_utils import get_lm_corpus
from hourglass import MemTransformerLM
from train import parse_args, gen_model_config, sample_generation, \
        evaluate, get_boundary_config
from utils.exp_utils import l2_promote
import pdb


def load_checkpoint(path):
    print(f'Starting path: {path}')
    path = os.path.dirname(path)
    while path.split('/')[-1] in ['configs', 'xl']:
        path = path = os.path.dirname(path)

    path = os.path.join(path, 'xl/LM-TFM/checkpoint_last.pt')

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
    boundary_kwargs = get_boundary_config(args)
    corpus = get_lm_corpus(args.data,
                           args.dataset,
                           **boundary_kwargs)
    vocab = corpus.vocab
    args.n_token = len(vocab)

    val_iter = corpus.get_iterator(split='valid',
                                  bsz=args.batch_size,
                                  tgt_len=2048,
                                  device=device,
                                  ext_len=0,
                                  **boundary_kwargs)

    print(args)

    world_size = utils.distributed.get_world_size()
    print(f'The world size is {world_size}')
    print(f'We expect {val_iter.n_batch} batches')

    data = [batch for batch in val_iter.get_fixlen_iter()]
    batch = data[0]
    input_data, target, seq_len, boundaries = batch
    input_data = input_data.cuda()
    target = target.cuda()
    boundaries = boundaries.cuda()

    ###########################################################################
    # Build the model
    ###########################################################################
    model_config = gen_model_config(args, vocab)
    model = MemTransformerLM(**model_config)
    model = model.to(device)

    ###########################################################################
    # Test
    ###########################################################################

    checkpoint = load_checkpoint(args.ckpt_path)
    if list(checkpoint['model_state'].keys())[0].startswith('module'):
        model_state = {
            k[7:]: v for k, v in checkpoint['model_state'].items()
        }
    else:
        model_state = checkpoint['model_state']

    model.load_state_dict(model_state)
    model.eval()

    target_test_len = 2048

    with torch.no_grad():
        loss, stats, _, boundaries_elem, entropy = model(
            data=input_data[:target_test_len, 3:4],
            target=target[:target_test_len, 3:4],
            boundaries=None,
            step=0,
        )
        loss = loss[:, 0]

    pdb.set_trace()

    boundaries_acc = []
    losses_acc = []
    stats_acc = []

    with torch.no_grad():
        for idx, (input_data, target, seq_len, boundaries) in enumerate(data):
            if idx > 5:
                break
            input_data = input_data.cuda()
            target = target.cuda()

            boundaries_in_acc = []

            for i in range(10):
                now_b = boundaries_in_acc[-1] if len(boundaries_in_acc) > 0 else None
                x = model(
                    data=input_data[:target_test_len, :4],
                    target=target[:target_test_len, :4],
                    boundaries=now_b,
                    step=0,
                )
                loss_elem, stats, _, boundaries_elem, entropy = x
                loss = loss_elem.cpu().detach()
                stats_acc.append(stats)
                losses_acc.append(loss)
                boundaries_in_acc.append(boundaries_elem)

            boundaries_acc.append(boundaries_in_acc)

    pdb.set_trace()

    sample_generation(vocab, val_iter.boundary_creator, model)

    if not args.debug and not args.no_eval:
        # Run on test data.
        test_start_time = time.time()
        with torch.autograd.profiler.emit_nvtx(enabled=args.profile):
            test_loss = evaluate(val_iter, model, args, step=200000)
            test_loss = utils.distributed.all_reduce_item(test_loss, 'mean')
        test_elapsed = time.time() - test_start_time
        print(f'Time elapsed: {test_elapsed}')


if __name__ == "__main__":
    main()
