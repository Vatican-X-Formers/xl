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
import numpy as np
import imageio as iio
import pdb
import glob
from torch.utils.data import DataLoader, Dataset
import utils
from utils.vocabulary import Vocab
from boundary_creator import get_boundary_creator


class LMOrderedIterator(object):
    def __init__(self, data, bsz, tgt_len, ext_len, vocab,
                 boundary_creator, **kwargs):
        """
            data -- LongTensor -- the LongTensor is strictly ordered
        """
        self.bsz = bsz
        self.tgt_len = tgt_len
        self.ext_len = ext_len if ext_len is not None else 0
        self.vocab = vocab

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

        assert boundary_creator is not None
        self.boundary_creator = boundary_creator
        self.boundaries = None
        if boundary_creator.boundaries_type in ['space_dist', 'normal']:
            print('Special case, for random boundaries we want to sample them once')
            self.boundaries = boundary_creator.get_boundaries(self.data).transpose(0, 1)

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
        batch_size = len(self.data)

        end_idx = i + seq_len
        beg_idx = max(0, i - self.ext_len)

        current_batch = [self.data[j][beg_idx:end_idx + 1] for j in range(len(self.data))]
        data = [self.vocab.convert_to_tensor(current_batch[j].replace(' ',
                                                                      '_')).unsqueeze(1) for j in range(batch_size)]
        data = torch.cat(data, dim=1).long().contiguous()
        target = data[-seq_len:]
        boundaries = self.boundary_creator.get_boundaries(txt=current_batch,
                                                          tensor=data)
        if boundaries is not None:
            boundaries = boundaries.t().bool().contiguous()[:-1, :]
        data = data[:-1, :]

        return data, target, seq_len, boundaries

    def get_fixlen_iter(self, start=0, shuffle=False):
        dataset = [i for i in range(start, self.data_len - 1, self.tgt_len)]

        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
            collate_fn=self.get_batch,
            num_workers=4
        )


class ImageDataset(Dataset):
    def __init__(self, filenames, bsz, **kwargs):
        self.filenames = filenames
        self.bsz = bsz

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
        stacked_batch = stacked_batch[:, :, :, 1]
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

        seq_len = stacked_batch.size(0)
        boundaries = None

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

        if dataset == 'cifar10':
            pass
        elif dataset == 'text8':
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

            for split in ['valid']:
                self.data[split] = [filename for filename in
                                    glob.glob(f'{path}{split}/*')]
            for split in ['train', 'test']:
                self.data[split] = self.data['valid']

    def extend_kwargs_for_bc(self, **kwargs):
        kwargs['boundary_ids'] = [self.vocab.sym2idx[c] for c in eval(kwargs['boundary_ids'])]
        return kwargs

    def get_iterator(self, split, **kwargs):
        if self.dataset == 'text8':
            kwargs = self.extend_kwargs_for_bc(**kwargs)
            return LMOrderedIterator(
                data=self.data[split],
                boundary_creator=get_boundary_creator(**kwargs),
                vocab=self.vocab,
                **kwargs
            )
        elif self.dataset in ['im32', 'cifar10']:
            return ImageDataset(filenames=self.data[split], **kwargs)


def get_lm_corpus(datadir, dataset, **kwargs):
    return Corpus(datadir, dataset, **kwargs)
