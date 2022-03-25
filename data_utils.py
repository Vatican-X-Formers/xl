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
import os
import torch
import numpy as np
import random
import pdb
from torch.utils.data import DataLoader

import utils
from utils.vocabulary import Vocab
from boundary_creator import get_boundary_checkpoint_name, get_boundary_creator

class LMOrderedIterator(object):
    def __init__(self, data, bsz, tgt_len, device='cpu', ext_len=None, boundary_creator=None):
        """
            data -- LongTensor -- the LongTensor is strictly ordered
        """
        self.bsz = bsz
        self.tgt_len = tgt_len
        self.ext_len = ext_len if ext_len is not None else 0

        self.device = device

        # Data is a tuple, the data tensor and boundaries from boundaries
        data, boundaries = data

        # Work out how cleanly we can divide the dataset into bsz parts.
        n_step = len(data) // bsz

        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data[:n_step * bsz]

        # Partition data for DistributedDataParallel
        world_size = utils.distributed.get_world_size()
        rank = utils.distributed.get_rank()

        assert len(data) % world_size == 0
        first_leap = len(data) // world_size
        data = [data[i:i + first_leap] for i in range(0, len(data), first_leap)]
        data = data[rank]
        data = [data[i:i + n_step] for i in range(0, len(data), n_step)]
        self.data = data

        if boundaries is not None:
            boundaries = boundaries[:n_step * bsz]
            self.boundaries = boundaries.view(bsz, -1).t().contiguous().pin_memory()
            self.boundaries = self.boundaries.chunk(world_size, dim=1)[rank]
        else:
            assert boundary_creator is not None
            self.boundaries = None
            if boundary_creator.boundaries_type in ['space_dist', 'normal']:
                print('Special case, for random boundaries we want to sample them once')
                self.boundaries = boundary_creator.get_boundaries(self.data).transpose(0, 1)

        self.boundary_creator = boundary_creator

        # Number of mini-batches
        self.data_len = len(data[0])
        self.n_batch = (self.data_len + self.tgt_len - 1) // self.tgt_len

        self.last_iter = None

    def roll(self, seed):
        raise NotImplementedError
        # rng = torch.Generator()
        # rng.manual_seed(seed)
        # for i in range(self.data.size(1)):
        #     row = self.data[:, i]
        #     shift = torch.randint(0, self.data_len, (1,), generator=rng)
        #     row = torch.cat((row[shift:], row[:shift]))
        #     self.data[:, i] = row

    def get_batch(self, i):
        i = i[0]
        seq_len = min(self.tgt_len, self.data_len - 1 - i)

        end_idx = i + seq_len
        beg_idx = max(0, i - self.ext_len)

        out = \
            self.boundary_creator.get_boundaries([self.data[i][beg_idx:end_idx + 1] for i in range(len(self.data))])
        data, target, boundaries = out
        n_examples = len(self.data)
        data = torch.tensor(np.concatenate(data)).reshape(n_examples,
                                                          -1).t().long()
        target = torch.tensor(np.concatenate(target)).reshape(n_examples,
                                                              -1).t().long()
        boundaries = torch.tensor(np.concatenate(boundaries)).reshape(n_examples, -1).t().bool()

        return data, target, seq_len, boundaries

    def get_fixlen_iter(self, start=0, shuffle=False):
        dataset = [i for i in range(start, self.data_len - 1, self.tgt_len)]

        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
            collate_fn=self.get_batch,
            num_workers=2
        )

    def __iter__(self):
        return self.get_fixlen_iter()


class Corpus(object):
    def __init__(self, path, dataset, *args, **kwargs):
        self.dataset = dataset
        self.data = {}
        self.vocab = [i for i in range(27)]

        for split in ['train', 'valid', 'test']:
            dataset_path = os.path.join(path, f'{split}.txt')
            sents = []
            with open(dataset_path, 'r', encoding='utf-8') as f:
                for idx, line in enumerate(f):
                    sents.append(line)
            assert len(sents) == 1
            sent = sents[0].replace(' ', '').replace('_', ' ')

            self.data[split] = sent, None

        kwargs = self.extend_kwargs_for_bc(**kwargs)
        self.boundary_creator = get_boundary_creator(**kwargs)

    def extend_kwargs_for_bc(self, **kwargs):
        kwargs['boundary_ids'] = []
        kwargs['vocab'] = []
        return kwargs

    def get_iterator(self, split, *args, **kwargs):
        kwargs = self.extend_kwargs_for_bc(**kwargs)
        return LMOrderedIterator(self.data[split], boundary_creator=get_boundary_creator(**kwargs), *args)


def get_lm_corpus(datadir, dataset, **kwargs):
    filename = get_boundary_checkpoint_name(datadir, **kwargs)
    logging.info(f'Target corpus is under path {filename}')
    corpus = Corpus(datadir, dataset, **kwargs)

    return corpus
