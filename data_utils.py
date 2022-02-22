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

import glob
import logging
import os
import re

import numpy as np
import torch

import utils
import pdb
from utils.vocabulary import Vocab
from boundary_creator import get_boundary_checkpoint_name, get_boundary_creator, TokenizerBoundaryCreator

class LMOrderedIterator(object):
    def __init__(self, data, bsz, tgt_len, device='cpu', ext_len=None, boundary_creator=None):
        """
            data -- LongTensor -- the LongTensor is strictly ordered
        """
        self.bsz = bsz
        self.tgt_len = tgt_len
        self.ext_len = ext_len if ext_len is not None else 0

        self.device = device

        # Data is most likely a tuple, the data tensor and boundaries from boundaries
        data, boundaries = data

        # Work out how cleanly we can divide the dataset into bsz parts.
        n_step = data.size(0) // bsz

        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data[:n_step * bsz]

        # Evenly divide the data across the bsz batches.
        self.data = data.view(bsz, -1).t().contiguous().pin_memory()

        # Partition data for DistributedDataParallel
        world_size = utils.distributed.get_world_size()
        rank = utils.distributed.get_rank()
        self.data = self.data.chunk(world_size, dim=1)[rank]

        if boundaries is not None:
            boundaries = boundaries[:n_step * bsz]
            self.boundaries = boundaries.view(bsz, -1).t().contiguous().pin_memory()
            self.boundaries = self.boundaries.chunk(world_size, dim=1)[rank]
        else:
            assert boundary_creator is not None
            self.boundaries = None

        self.boundary_creator = boundary_creator

        # Number of mini-batches
        self.n_batch = (self.data.size(0) + self.tgt_len - 1) // self.tgt_len

        self.last_iter = None

    def roll(self, seed):
        rng = torch.Generator()
        rng.manual_seed(seed)
        for i in range(self.data.size(1)):
            row = self.data[:, i]
            shift = torch.randint(0, self.data.size(0), (1,), generator=rng)
            row = torch.cat((row[shift:], row[:shift]))
            self.data[:, i] = row

    def get_batch(self, i, tgt_len=None):
        if tgt_len is None:
            tgt_len = self.tgt_len

        seq_len = min(tgt_len, self.data.size(0) - 1 - i)

        end_idx = i + seq_len
        beg_idx = max(0, i - self.ext_len)

        data = self.data[beg_idx:end_idx].to(self.device, non_blocking=True)
        if self.boundaries is not None:
            boundaries = self.boundaries[beg_idx:end_idx].to(self.device, non_blocking=True)
        else:
            boundaries = self.boundary_creator.get_boundaries(data)
            if boundaries is not None:
                print('not none')
                boundaries = boundaries.transpose(0, 1)
        target = self.data[i+1:i+1+seq_len].to(self.device, non_blocking=True)

        return data, target, seq_len, boundaries

    def get_fixlen_iter(self, start=0):
        if start != 0:
            start += self.tgt_len
        for i in range(start, self.data.size(0) - 1, self.tgt_len):
            self.last_iter = i
            yield self.get_batch(i)

    def __iter__(self):
        return self.get_fixlen_iter()


class Corpus(object):
    def __init__(self, path, dataset, *args, **kwargs):
        self.dataset = dataset
        self.vocab = Vocab(*args, **kwargs)
        self.data = {}

        for split in ['train', 'valid', 'test']:
            dataset_path = os.path.join(path, f'{split}.txt')
            self.vocab.count_file(dataset_path)
        
        self.vocab.build_vocab()
        boundary_ids = [self.vocab.sym2idx[c] for c in eval(kwargs['boundary_ids'])]
        kwargs['boundary_ids'] = boundary_ids
        
        self.boundary_creator = get_boundary_creator(**kwargs)
        extract_boundaries = self.boundary_creator.extract_offline

        for split in ['train', 'valid', 'test']:
            dataset_path = os.path.join(path, f'{split}.txt')
            if self.dataset in ['enwik8']:
                self.data[split] = self.vocab.encode_file(
                        dataset_path, 
                        add_eos=True,
                        boundary_creator=self.boundary_creator,
                        extract_boundaries=extract_boundaries,
                )
            elif self.dataset in ['text8']:
                self.data[split] = self.vocab.encode_file(
                        dataset_path, 
                        add_eos=False,
                        boundary_creator=self.boundary_creator,
                        extract_boundaries=extract_boundaries,
                )

    def get_iterator(self, split, *args, **kwargs):
        boundary_ids = [self.vocab.sym2idx[c] for c in eval(kwargs['boundary_ids'])]
        kwargs['boundary_ids'] = boundary_ids
        return LMOrderedIterator(self.data[split], boundary_creator=get_boundary_creator(**kwargs), *args)


def get_lm_corpus(datadir, dataset, **kwargs):
    filename = get_boundary_checkpoint_name(datadir, kwargs['boundaries_type'], kwargs['boundaries_tokens'])
    logging.info(f'Target corpus is under path {filename}')

    if os.path.exists(filename):
        logging.info('\nLoading cached dataset...')
        corpus = torch.load(filename)
    else:
        logging.info('\nProducing dataset {}...'.format(dataset))
        if dataset == 'enwik8':
            kwargs['special'] = ['<eos>']
        elif dataset == 'text8':
            pass

        corpus = Corpus(datadir, dataset, **kwargs)
        with utils.distributed.sync_workers() as rank:
            if rank == 0:
                torch.save(corpus, filename)

    return corpus
