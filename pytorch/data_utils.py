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
from utils.vocabulary import OpenAIVocab
from utils.vocabulary import Vocab


class LMOrderedIterator(object):
    def __init__(self, data, bsz, bptt, device='cpu', mem_len=None, ext_len=None, warmup=True):
        """
            data -- LongTensor -- the LongTensor is strictly ordered
        """
        self.bsz = bsz
        self.bptt = bptt
        self.ext_len = ext_len if ext_len is not None else 0
        self.mem_len = mem_len
        self.warmup = warmup

        self.device = device

        # Work out how cleanly we can divide the dataset into bsz parts.
        n_step = data.size(0) // bsz

        # print('in data iterator')
        # print(len(data))
        # print(data.size())

        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data[:n_step * bsz]

        # Evenly divide the data across the bsz batches.
        self.data = data.view(bsz, -1).t().contiguous().pin_memory()

        if mem_len and warmup:
            self.warmup_batches = (mem_len + bptt - 1) // bptt
            self.warmup_elems = self.warmup_batches * bptt

            warmup_data = self.data.roll((self.warmup_elems, 1), (0, 1))[:self.warmup_elems]
            self.data = torch.cat((warmup_data, self.data))

        # Partition data for DistributedDataParallel
        world_size = utils.distributed.get_world_size()
        rank = utils.distributed.get_rank()
        self.data = self.data.chunk(world_size, dim=1)[rank]

        # Number of mini-batches
        self.n_batch = (self.data.size(0) + self.bptt - 1) // self.bptt

        # So how this shit work
        # I start with the stream of tokens, lets say 6k tokens
        # I say that my batch_size is 10 and target_len is 60
        # So I expect mini_batches of size (10, 60)
        # Normally I could just divide 6k into chunks of 60 and then feed it to batcher
        # But then I wouldn't use memory at all because I would 
        # be processing consecutive elems in the same batch
        # The solution here is to divide data onto batches
        # So we get very big stripes at next batches if not shuffled would depened on previos one
        # And there would be no dependency inside batches

        # Robią jakieś kilka batchy warm-up'u
        # Nie robią shift right w tym modelu, shift right jest robiony przy podawaniu danych
        # Co to znaczy, znaczy to to, ze target jest przesuniety o jeden w prawo i jest 
        # Zapewnienione ze target zawsze istnieje. Bierzemy input dla ktorego znamy kolejny znak,
        # Więc target zawsze istnieje. Będę musiał dostosować funnel'a pod te specyfikacje zadania

        # Ext len jest to dodatkowe ustawienie data loadera, ktore pozwala na feedowanie dluzszych inputow
        # The thing that happens with data loader ext_len is that we make jumps over data that are tgt_len
        # But we feed ext_len to the model actually
        # So not only we use memory but we only use this last steps silently incorporated into model parameters

        # Nie da się zagwarantować, ze input będzie podzielny przez sf, więc to będę musiał ogarnąć, jak w w2v

        self.last_iter = None

    def roll(self, seed):
        rng = torch.Generator()
        rng.manual_seed(seed)
        for i in range(self.data.size(1)):
            row = self.data[:, i]
            shift = torch.randint(0, self.data.size(0), (1,), generator=rng)
            row = torch.cat((row[shift:], row[:shift]))
            self.data[:, i] = row

    def get_batch(self, i, bptt=None):
        if bptt is None:
            bptt = self.bptt

        seq_len = min(bptt, self.data.size(0) - 1 - i)

        end_idx = i + seq_len
        beg_idx = max(0, i - self.ext_len)

        data = self.data[beg_idx:end_idx].to(self.device, non_blocking=True)
        target = self.data[i+1:i+1+seq_len].to(self.device, non_blocking=True)

        if self.mem_len and self.warmup:
            warm = i >= self.warmup_elems
        else:
            warm = True

        return data, target, seq_len, warm

    def get_fixlen_iter(self, start=0):
        if start != 0:
            start += self.bptt
        for i in range(start, self.data.size(0) - 1, self.bptt):
            self.last_iter = i
            yield self.get_batch(i)

    def __iter__(self):
        return self.get_fixlen_iter()


class Corpus(object):
    def __init__(self, path, dataset, vocab, *args, **kwargs):
        self.dataset = dataset
        if vocab == 'word':
            self.vocab = Vocab(*args, **kwargs)
        elif vocab == 'bpe':
            self.vocab = OpenAIVocab()
        else:
            raise RuntimeError('Unsupported vocab')

        if self.dataset in ['ptb', 'wt2', 'enwik8', 'text8']:
            self.vocab.count_file(os.path.join(path, 'train.txt'))
            self.vocab.count_file(os.path.join(path, 'valid.txt'))
            self.vocab.count_file(os.path.join(path, 'test.txt'))
        elif self.dataset == 'wt103':
            self.vocab.count_file(os.path.join(path, 'train.txt'))
        elif self.dataset == 'lm1b':
            train_path_pattern = os.path.join(
                path, '1-billion-word-language-modeling-benchmark-r13output',
                'training-monolingual.tokenized.shuffled', 'news.en-*')
            train_paths = glob.glob(train_path_pattern)
            # the vocab will load from file when build_vocab() is called

        self.vocab.build_vocab()

        if self.dataset in ['ptb', 'wt2', 'wt103']:
            self.train = self.vocab.encode_file(
                os.path.join(path, 'train.txt'), ordered=True)
            self.valid = self.vocab.encode_file(
                os.path.join(path, 'valid.txt'), ordered=True)
            self.test = self.vocab.encode_file(
                os.path.join(path, 'test.txt'), ordered=True)
        elif self.dataset in ['enwik8']:
            self.train = self.vocab.encode_file(
                os.path.join(path, 'train.txt'), ordered=True, add_eos=True)
            self.valid = self.vocab.encode_file(
                os.path.join(path, 'valid.txt'), ordered=True, add_eos=True)
            self.test = self.vocab.encode_file(
                os.path.join(path, 'test.txt'), ordered=True, add_eos=True)
        elif self.dataset in ['text8']:
            # There are no linebreaks on text8
            self.train = self.vocab.encode_file(
                os.path.join(path, 'train.txt'), ordered=True, add_eos=False)
            self.valid = self.vocab.encode_file(
                os.path.join(path, 'valid.txt'), ordered=True, add_eos=False)
            self.test = self.vocab.encode_file(
                os.path.join(path, 'test.txt'), ordered=True, add_eos=False)
        elif self.dataset == 'lm1b':
            self.train = train_paths
            self.valid = self.vocab.encode_file(
                os.path.join(path, 'valid.txt'), ordered=False, add_double_eos=True)
            self.test = self.vocab.encode_file(
                os.path.join(path, 'test.txt'), ordered=False, add_double_eos=True)

    def get_iterator(self, split, *args, **kwargs):
        if split == 'train':
            data_iter = LMOrderedIterator(self.train, *args, **kwargs)
        elif split in ['valid', 'test']:
            data = self.valid if split == 'valid' else self.test
            data_iter = LMOrderedIterator(data, *args, **kwargs)

        return data_iter


def get_lm_corpus(datadir, dataset, vocab):
    if vocab == 'word':
        fn = os.path.join(datadir, 'cache.pt')
    elif vocab == 'bpe':
        fn = os.path.join(datadir, 'cache.pt.bpe')
    else:
        raise RuntimeError('Unsupported vocab')

    if os.path.exists(fn):
        logging.info('Loading cached dataset...')
        corpus = torch.load(fn)
    else:
        logging.info('Producing dataset {}...'.format(dataset))
        kwargs = {}
        if dataset in ['wt103', 'wt2']:
            kwargs['special'] = ['<eos>']
            kwargs['lower_case'] = False
        elif dataset == 'ptb':
            kwargs['special'] = ['<eos>']
            kwargs['lower_case'] = True
        elif dataset == 'lm1b':
            kwargs['special'] = []
            kwargs['lower_case'] = False
            kwargs['vocab_file'] = os.path.join(datadir, '1b_word_vocab.txt')
        elif dataset == 'enwik8':
            # We preserve eos on enwiki
            kwargs['special'] = ['<eos>']
        elif dataset == 'text8':
            # There are no eos on text8
            pass

        corpus = Corpus(datadir, dataset, vocab, **kwargs)
        with utils.distributed.sync_workers() as rank:
            if rank == 0:
                torch.save(corpus, fn)

    return corpus
