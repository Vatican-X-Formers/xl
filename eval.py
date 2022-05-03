import os
import time

import numpy as np
import torch

import utils
from data_utils import get_lm_corpus
from hourglass import MemTransformerLM
from train import parse_args, gen_model_config, sample_generation, \
        evaluate, get_boundary_config, get_logits
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


def autoregressive_test(data_full, boundaries_full, boundary_creator, model,
                        steps=30, bs=2):
    model.eval()

    # data is of size time_steps x bs
    data_full = data_full[:steps, :bs]
    if boundaries_full is not None:
        boundaries_full = boundaries_full[:steps, :bs]

    with torch.no_grad():
        logits = get_logits(model, data_full, boundaries_full)

        for i in range(2, steps + 1, 1):
            partial_data = data_full[:i]
            partial_boundaries = boundary_creator.get_boundaries(
                txt=None,
                tensor=partial_data,
            )
            if partial_boundaries is not None:
                partial_boundaries = partial_boundaries.t().bool().contiguous()
            logits_partial = get_logits(model, partial_data, partial_boundaries)
            assert torch.allclose(logits_partial, logits[:i], rtol=1e-5, atol=1e-5)

    print('Passed autoregressive test')

    model.train()


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

    # TODO OVERRIDE
    args.batch_size = 4

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
    if boundaries is not None:
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
        if 'boundary_predictor.loss.weight' in model_state:
            del model_state['boundary_predictor.loss.weight']
    else:
        model_state = checkpoint['model_state']

    model.load_state_dict(model_state)
    model.eval()

    pdb.set_trace()

    autoregressive_test(input_data, boundaries, val_iter.boundary_creator, model, steps=30)
    import sys
    sys.exit()

    def calc_entropy(logit):
        entropy = -torch.nn.functional.log_softmax(logit, dim=-1) * torch.nn.functional.softmax(logit, dim=-1)
        entropy = torch.sum(entropy, dim=-1)
        return entropy

    with torch.no_grad():
        for i in range(5):
            input_data, target, seq_len, boundaries = data[i]
            input_data = input_data.cuda()
            target = target.cuda()
            if boundaries is not None:
                boundaries = boundaries.cuda()

            l_idx = i * 300 + 100
            input_data = input_data[l_idx:l_idx + 120, :1]

            text = vocab.convert_to_sent(input_data[:, 0], mode='real')

            logits = model(
                input_data,
                None,
                None,
                None,
                0
            )

            entropy = calc_entropy(logits)

            print(text)
            print(entropy[:, 0].cpu())

    import sys
    sys.exit()

    boundaries_acc = []
    losses_acc = []

    with torch.no_grad():
        for i in range(30):
            print(i)
            input_data, target, seq_len, boundaries = data[i]

            input_data = input_data.cuda()
            target = target.cuda()
            if boundaries is not None:
                boundaries = boundaries.cuda()

            # loss, _, _, target_bp_mask = model(
            #     input_data,
            #     None,
            #     None, None, 0
            # )
            logits = model(
                input_data,
                None,
                None, None, 0
            )

            losses_acc.append(logits)

    xd = torch.cat(losses_acc, dim=1)
    torch.save(xd, 'char.pt')

    import sys
    sys.exit(0)

    target_test_len = 2048
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
