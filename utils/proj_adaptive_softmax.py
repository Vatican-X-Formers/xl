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

import torch
import torch.nn as nn
import torch.nn.functional as F


class OptionalParameterList(nn.ParameterList):
    def extra_repr(self):
        child_lines = []
        for k, p in self._parameters.items():
            if p is not None:
                size_str = 'x'.join(str(size) for size in p.size())
                device_str = '' if not p.is_cuda else ' (GPU {})'.format(p.get_device())
                parastr = 'Parameter containing: [{} of size {}{}]'.format(
                    torch.typename(p), size_str, device_str)
                child_lines.append('  (' + str(k) + '): ' + parastr)
        tmpstr = '\n'.join(child_lines)
        return tmpstr


class ProjectedAdaptiveLogSoftmax(nn.Module):
    def __init__(self, n_token, d_embed, d_proj, cutoffs, div_val=1,
                 tie_projs=None, out_layers_weights=None, out_projs=None,
                 keep_order=False):
        super().__init__()

        self.n_token = n_token
        self.d_embed = d_embed
        self.d_proj = d_proj

        self.cutoffs = cutoffs + [n_token]
        self.cutoff_ends = [0] + self.cutoffs
        self.div_val = div_val

        self.shortlist_size = self.cutoffs[0]
        self.n_clusters = len(self.cutoffs) - 1
        self.head_size = self.shortlist_size + self.n_clusters

        self.tie_projs = tie_projs

        if self.n_clusters > 0:
            self.cluster_weight = nn.Parameter(torch.zeros(self.n_clusters, self.d_embed))
            self.cluster_bias = nn.Parameter(torch.zeros(self.n_clusters))

        if not out_layers_weights:
            self.out_layers_weights = nn.ParameterList()
        else:
            self.out_layers_weights = out_layers_weights

        self.out_layers_biases = nn.ParameterList()

        self.shared_out_projs = out_projs
        self.out_projs = OptionalParameterList()

        if div_val == 1:
            if d_proj != d_embed:
                for i in range(len(self.cutoffs)):
                    if tie_projs[i]:
                        self.out_projs.append(None)
                    else:
                        self.out_projs.append(
                            nn.Parameter(torch.zeros(d_proj, d_embed))
                        )
            else:
                # self.out_projs = [None] * len(self.cutoffs)
                self.out_projs.append(None)

            self.out_layers_biases.append(
                nn.Parameter(torch.zeros(n_token))
                )

            if not out_layers_weights:
                self.out_layers_weights.append(
                    nn.Parameter(torch.zeros(n_token, d_embed))
                    )
        else:
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i+1]
                d_emb_i = d_embed // (div_val ** i)

                if tie_projs[i]:
                    self.out_projs.append(None)
                else:
                    self.out_projs.append(
                        nn.Parameter(torch.zeros(d_proj, d_emb_i))
                    )

                self.out_layers_biases.append(
                    nn.Parameter(torch.zeros(r_idx - l_idx))
                    )
                if not out_layers_weights:
                    self.out_layers_weights.append(
                        nn.Parameter(torch.zeros(r_idx - l_idx, d_emb_i))
                        )

        self.keep_order = keep_order

    def _compute_logit(self, hidden, weight, bias, proj):
        logit = F.linear(hidden, weight, bias=bias)
        return logit

    def get_out_proj(self, i):
        if self.tie_projs[i]:
            if len(self.shared_out_projs) == 0:
                return None
            elif len(self.shared_out_projs) == 1:
                return self.shared_out_projs[0]
            else:
                return self.shared_out_projs[i]
        else:
            return self.out_projs[i]

    def forward(self, hidden, target, keep_order=False):
        '''
            hidden :: [len*bsz x d_proj]
            target :: [len*bsz]
        '''
        nll = -F.log_softmax(logit, dim=-1).gather(1, target.unsqueeze(1)).squeeze(1)
        return nll
