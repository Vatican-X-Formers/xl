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

from collections import Counter
from collections import OrderedDict

import torch

class Vocab(object):
    def __init__(self, special=[], min_freq=0, max_size=None, lower_case=True,
                 delimiter=None, boundary_creator=None, extract_boundaries=False,
                 **kwargs):
        self.counter = Counter()
        self.special = special
        self.min_freq = min_freq
        self.max_size = max_size
        self.lower_case = lower_case
        self.delimiter = delimiter
        self.boundary_creator = boundary_creator
        self.extract_boundaries = extract_boundaries

    def build_vocab(self):
        print('building vocab with min_freq={}, max_size={}'.format(
            self.min_freq, self.max_size))
        self.idx2sym = []
        self.sym2idx = OrderedDict()

        for sym in self.special:
            self.add_special(sym)

        for sym, cnt in self.counter.most_common(self.max_size):
            if cnt < self.min_freq:
                break
            self.add_symbol(sym)

        assert ((self.idx2sym[0] == '_') or (self.idx2sym[0] == ' ')), 'first symbol is not a space'

        print('final vocab size {} from {} unique tokens'.format(
            len(self), len(self.counter)))

    def add_special(self, sym):
        if sym not in self.sym2idx:
            self.idx2sym.append(sym)
            self.sym2idx[sym] = len(self.idx2sym) - 1
            setattr(self, '{}_idx'.format(sym.strip('<>')), self.sym2idx[sym])

    def add_symbol(self, sym):
        if sym not in self.sym2idx:
            self.idx2sym.append(sym)
            self.sym2idx[sym] = len(self.idx2sym) - 1

    def get_sym(self, idx):
        assert 0 <= idx < len(self), 'Index {} out of range'.format(idx)
        return self.idx2sym[idx]

    def get_idx(self, sym):
        if sym in self.sym2idx:
            return self.sym2idx[sym]
        else:
            assert '<eos>' not in sym
            assert hasattr(self, 'unk_idx')
            return self.sym2idx.get(sym, self.unk_idx)

    def get_symbols(self, indices):
        return [self.get_sym(idx) for idx in indices]

    def get_indices(self, symbols):
        return [self.get_idx(sym) for sym in symbols]

    def convert_to_tensor(self, symbols):
        return torch.LongTensor(self.get_indices(symbols))

    def convert_to_sent(self, indices, mode='dataset'):
        # Mode is either:
        # - 'dataset' = It means that we get the dataset in the special format
        # where each character is delimited by whitespaces and original
        # whitespaces are usually encoded as some other special character
        # - 'real' = It looks like a real sentence
        if mode == 'dataset':
            return ' '.join([self.get_sym(idx) for idx in indices])
        else:
            return ''.join([self.get_sym(idx) for idx in indices]).replace('_', ' ')

    def __len__(self):
        return len(self.idx2sym)

