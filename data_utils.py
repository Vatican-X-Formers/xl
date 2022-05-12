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

import os
import torch
import imageio as iio
import pdb
import random
from torch.utils.data import DataLoader, Dataset
import utils
from utils.vocabulary import Vocab
from boundary_creator import get_boundary_creator


class LMOrderedIterator(object):
    def __init__(self, data, bsz, tgt_len, ext_len, vocab,
                 boundary_creator, dataset, **kwargs):
        """
            data -- LongTensor -- the LongTensor is strictly ordered
        """
        self.bsz = bsz
        self.tgt_len = tgt_len
        self.ext_len = ext_len if ext_len is not None else 0
        self.vocab = vocab
        self.dataset = dataset

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
        self.txt = data
        self.data = torch.cat([self.vocab.convert_to_tensor(self.txt[j]).unsqueeze(-1)
                               for j in range(len(self.txt))], dim=1)

        self.boundary_creator = boundary_creator
        self.boundaries = boundary_creator.get_boundaries(txt=self.txt).bool().transpose(0, 1).contiguous()
        self.data = self.data.cuda()
        self.boundaries = self.boundaries.cuda()

        # Number of mini-batches
        self.data_len = len(data[0])
        self.n_batch = (self.data_len + self.tgt_len - 1) // self.tgt_len

        self.last_iter = None
        self.device = kwargs['device']

    def roll(self, seed):
        rng = torch.Generator()
        rng.manual_seed(seed)
        for i in range(len(self.data)):
            row = self.data[i]
            shift = torch.randint(0, self.data_len, (1,), generator=rng)
            row = row[shift:] + row[:shift]
            self.data[i] = row

    def get_batch(self, i):
        i = i[0]
        seq_len = min(self.tgt_len, self.data_len - 1 - i)

        end_idx = i + seq_len
        beg_idx = max(0, i - self.ext_len)

        data = self.data[beg_idx:end_idx + 1]
        target = data[-seq_len:]
        data = data[:-1]
        boundaries = self.boundaries[beg_idx:end_idx]

        return data, target, seq_len, boundaries

    def get_fixlen_iter(self, start=0, shuffle=False, seed=None, nw=0):
        dataset = [i for i in range(start, self.data_len - 1, self.tgt_len)]

        if shuffle:
            assert seed is not None
            random.seed(seed)
            random.shuffle(dataset)

        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
            collate_fn=self.get_batch,
            num_workers=nw
        )


class ImageDataset(Dataset):
    def __init__(self, filenames, bsz, boundary_creator, **kwargs):
        self.filenames = filenames
        self.bsz = bsz
        self.boundary_creator = boundary_creator

        world_size = utils.distributed.get_world_size()
        rank = utils.distributed.get_rank()

        assert self.bsz % world_size == 0 and self.bsz >= world_size
        self.n_batch = len(self.filenames) // self.bsz

        # Discard some leftovers so that # of examples is divisible by bsz
        self.filenames = self.filenames[:self.n_batch * self.bsz]
        self.device = kwargs['device']

        # MultiGPU dataloader
        self.filenames = self.filenames[rank::world_size]
        self.n_examples = len(self.filenames)
        self.local_bsz = self.bsz // world_size

    def unpickle(file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict[b'data']

    def read_image(self, filename):
        return iio.imread(filename)

    def __len__(self):
        return self.n_examples

    def __getitem__(self, idx):
        image = self.read_image(self.filenames[idx])
        return torch.tensor(image)

    def get_batch(self, batch):
        stacked_batch = torch.stack(batch)
        stacked_batch = stacked_batch.reshape(stacked_batch.size(0), -1)
        stacked_batch = stacked_batch.t().long()
        target = stacked_batch
        input = torch.cat(
            [
                torch.full(size=(1, target.size(-1)), fill_value=256,
                           device=target.device, dtype=torch.long),
                target[:-1],
            ]
        )
        boundaries = self.boundary_creator.get_boundaries(txt=None,
                                                          tensor=input)
        if boundaries is not None:
            boundaries = boundaries.t().bool().contiguous()

        target = target.contiguous()
        input = input.contiguous()

        seq_len = stacked_batch.size(0)

        return input, target, seq_len, boundaries

    def get_fixlen_iter(self, start=None, shuffle=False):
        return DataLoader(
            self,
            batch_size=self.local_bsz,
            shuffle=False,
            pin_memory=True,
            collate_fn=self.get_batch,
            num_workers=4
        )


class Corpus(object):
    def __init__(self, path, dataset, *args, **kwargs):
        self.dataset = dataset
        self.data = {}

        if dataset == 'text8':
            self.vocab = Vocab(*args, **kwargs)
            for split in ['train', 'valid', 'test']:
                dataset_path = os.path.join(path, f'{split}.txt')
                self.vocab.count_file(dataset_path)
                sents = []
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    for idx, line in enumerate(f):
                        sents.append(line)
                assert len(sents) == 1
                sent = sents[0].replace(' ', '').replace('_', ' ')

                self.data[split] = sent

            self.vocab.build_vocab()
        elif dataset == 'im32':
            self.vocab = [i for i in range(256)]

            for split in ['train', 'valid']:
                self.data[split] = []
                with open(f'{path}{split}.txt', 'r+') as file:
                    for line in file:
                        self.data[split].append(f'{path}{line.strip()}')

            for split in ['test']:
                self.data[split] = self.data['valid']
        elif dataset.startswith('wiki40b'):
            self.vocab = Vocab(*args, **kwargs)
            for split in ['valid', 'test']:
                dataset_path = os.path.join(path, f'{split}.txt')
                sents = []
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    for idx, line in enumerate(f):
                        sents.append(line)
                assert len(sents) == 1
                sent = sents[0]
                self.vocab.counter.update(sent)
                self.data[split] = sent
            self.data['train'] = self.data['valid']

            self.vocab.build_vocab()

    def extend_kwargs_for_bc(self, **kwargs):
        kwargs['boundary_ids'] = [self.vocab.sym2idx[c] for c in eval(kwargs['boundary_ids'])]
        kwargs['dataset'] = self.dataset
        return kwargs

    def get_iterator(self, split, **kwargs):
        if self.dataset in ['text8'] or self.dataset.startswith('wiki40b'):
            kwargs = self.extend_kwargs_for_bc(**kwargs)
            return LMOrderedIterator(
                data=self.data[split],
                boundary_creator=get_boundary_creator(**kwargs),
                vocab=self.vocab,
                **kwargs
            )
        elif self.dataset in ['im32', 'cifar10']:
            return ImageDataset(
                filenames=self.data[split],
                boundary_creator=get_boundary_creator(**kwargs),
                **kwargs
            )


def get_lm_corpus(datadir, dataset, **kwargs):
    return Corpus(datadir, dataset, **kwargs)
