import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.proj_adaptive_softmax import ProjectedAdaptiveLogSoftmax


@torch.jit.script
def add_and_scale(tensor1, tensor2, alpha: float):
    return alpha * (tensor1 + tensor2)


class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[:, None, :].expand(-1, bsz, -1)
        else:
            return pos_emb[:, None, :]


class PositionwiseFF(nn.Module):
    def __init__(self, d_model, d_inner, dropout, pre_lnorm=False):
        super(PositionwiseFF, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        self.CoreNet = nn.Sequential(
            nn.Linear(d_model, d_inner), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )

        self.layer_norm = nn.LayerNorm(d_model)

        self.pre_lnorm = pre_lnorm

    def forward(self, inp):
        if self.pre_lnorm:
            # layer normalization + positionwise feed-forward
            core_out = self.CoreNet(self.layer_norm(inp))

            # residual connection
            output = core_out + inp
        else:
            # positionwise feed-forward
            core_out = self.CoreNet(inp)

            # residual connection + layer normalization
            output = self.layer_norm(inp + core_out)

        return output


class RelMultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                 tgt_len=None, ext_len=None, mem_len=None, pre_lnorm=False):
        super(RelMultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

    def _parallelogram_mask(self, h, w, left=False):
        mask = torch.ones((h, w)).byte()
        m = min(h, w)
        mask[:m, :m] = torch.triu(mask[:m, :m])
        mask[-m:, -m:] = torch.tril(mask[-m:, -m:])

        if left:
            return mask.bool()
        else:
            return mask.flip(0).bool()

    def _shift(self, x, qlen, klen, mask, left=False):
        if qlen > 1:
            zero_pad = torch.zeros((x.size(0), qlen-1, x.size(2), x.size(3)),
                                   device=x.device, dtype=x.dtype)
        else:
            zero_pad = torch.zeros(0, device=x.device, dtype=x.dtype)

        if left:
            mask = mask.flip(1)
            x_padded = torch.cat([zero_pad, x], dim=1).expand(qlen, -1, -1, -1)
        else:
            x_padded = torch.cat([x, zero_pad], dim=1).expand(qlen, -1, -1, -1)

        x = x_padded.masked_select(mask[:, :, None, None]) \
                    .view(qlen, klen, x.size(2), x.size(3))

        return x

    def _rel_shift(self, x, zero_triu=False):
        zero_pad = torch.zeros((x.size(0), x.size(1), x.size(2), 1),
                               device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=3)

        x_padded = x_padded.view(x.size(0), x.size(1), x.size(3) + 1, x.size(2))

        x = x_padded.narrow(2, 1, x_padded.size(2) - 1).view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(2), x.size(3)))
            x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]

        return x

    def forward(self, w, r, attn_mask=None, mems=None):
        raise NotImplementedError


class RelPartialLearnableMultiHeadAttn(RelMultiHeadAttn):
    def __init__(self, *args, **kwargs):
        super(RelPartialLearnableMultiHeadAttn, self).__init__(*args, **kwargs)

        self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head, bias=False)

    def forward(self, w, r, r_w_bias, r_r_bias, attn_mask=None, mems=None):
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)

        if mems is not None:
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)  # qlen x bsz x n_head x d_head
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)  # klen x bsz x n_head x d_head
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)  # klen x bsz x n_head x d_head

        r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)       # qlen x n_head x d_head

        # compute attention score
        rw_head_q = w_head_q + r_w_bias                                # qlen x bsz x n_head x d_head
        AC = torch.einsum('ibnd,jbnd->bnij', rw_head_q, w_head_k)      # bsz x n_head x qlen x klen

        rr_head_q = w_head_q + r_r_bias
        BD = torch.einsum('ibnd,jnd->bnij', rr_head_q, r_head_k)       # bsz x n_head x qlen x klen
        BD = self._rel_shift(BD)

        # [bsz x n_head x qlen x klen]
        attn_score = add_and_scale(AC, BD, self.scale)

        # compute attention probability
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_score.masked_fill_(attn_mask[None, None, :, :], -float('inf'))
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:, None, :, :], -float('inf'))

        # [bsz x n_head x qlen x klen]
        attn_prob = F.softmax(attn_score, dim=3)
        attn_prob = self.dropatt(attn_prob)

        # compute attention vector
        attn_vec = torch.einsum('bnij,jbnd->ibnd', attn_prob, w_head_v)

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        # linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            # residual connection
            output = w + attn_out
        else:
            # residual connection + layer normalization
            output = self.layer_norm(w + attn_out)

        return output


class RelPartialLearnableDecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout,
                 **kwargs):
        super(RelPartialLearnableDecoderLayer, self).__init__()

        self.dec_attn = RelPartialLearnableMultiHeadAttn(n_head, d_model,
                                                         d_head, dropout,
                                                         **kwargs)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout,
                                     pre_lnorm=kwargs.get('pre_lnorm'))

    def forward(self, dec_inp, r, r_w_bias, r_r_bias, dec_attn_mask=None, mems=None):

        output = self.dec_attn(dec_inp, r, r_w_bias, r_r_bias,
                               attn_mask=dec_attn_mask,
                               mems=mems)
        output = self.pos_ff(output)

        return output


class AdaptiveEmbedding(nn.Module):
    def __init__(self, n_token, d_embed, d_proj, cutoffs, div_val=1,
                 sample_softmax=False):
        super(AdaptiveEmbedding, self).__init__()

        self.n_token = n_token
        self.d_embed = d_embed

        self.cutoffs = cutoffs + [n_token]
        self.div_val = div_val
        self.d_proj = d_proj

        self.emb_scale = d_proj ** 0.5

        self.cutoff_ends = [0] + self.cutoffs

        self.emb_layers = nn.ModuleList()
        self.emb_projs = nn.ParameterList()
        if div_val == 1:
            self.emb_layers.append(
                nn.Embedding(n_token, d_embed, sparse=(sample_softmax > 0))
            )
            if d_proj != d_embed:
                self.emb_projs.append(nn.Parameter(torch.Tensor(d_proj, d_embed).zero_()))
        else:
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i+1]
                d_emb_i = d_embed // (div_val ** i)
                self.emb_layers.append(nn.Embedding(r_idx-l_idx, d_emb_i))
                self.emb_projs.append(nn.Parameter(torch.Tensor(d_proj, d_emb_i).zero_()))

    def forward(self, inp):
        if self.div_val == 1:
            embed = self.emb_layers[0](inp)
            if self.d_proj != self.d_embed:
                embed = F.linear(embed, self.emb_projs[0])
        else:
            param = next(self.parameters())
            inp_flat = inp.view(-1)
            emb_flat = torch.zeros([inp_flat.size(0), self.d_proj],
                                   dtype=param.dtype, device=param.device)
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]

                mask_i = (inp_flat >= l_idx) & (inp_flat < r_idx)
                indices_i = mask_i.nonzero(as_tuple=False).squeeze()

                if indices_i.numel() == 0:
                    continue

                inp_i = inp_flat.index_select(0, indices_i) - l_idx
                emb_i = self.emb_layers[i](inp_i)
                emb_i = F.linear(emb_i, self.emb_projs[i]).to(emb_flat.dtype)

                emb_flat.index_copy_(0, indices_i, emb_i)

            embed = emb_flat.view(*inp.size(), self.d_proj)

        embed.mul_(self.emb_scale)

        return embed


class Upsampler(nn.Module):
    def __init__(self, embedding_dim, upsample_factor, mode='linear'):
        super().__init__()
        self.upsample_factor = upsample_factor
        self.mode = mode
        if mode == 'linear':
            self.upsample_layer = nn.Linear(embedding_dim, embedding_dim * upsample_factor)
    
    def forward(self, x, residual):
        # T x B x C
        if self.mode == 'linear':
            # T x B x C -> B x T x C
            x = x.transpose(0, 1)
            x = self.upsample_layer(x)
            x = x.reshape(x.size(0), x.size(1) * self.upsample_factor, -1)
            x = x.transpose(0, 1)
        else:
            x = x.repeat_interleave(self.upsample_factor, dim=0)

        assert x.size(0) >= residual.size(0)
        x = x[:residual.size(0)] + residual

        return x


class Downsampler(nn.Module):
    def __init__(self, embedding_dim, downsample_factor, mode='linear'):
        super().__init__()
        self.mode = mode
        self.downsample_factor = downsample_factor
        if mode == 'linear':
            self.downsample_layer = nn.Linear(
                embedding_dim * downsample_factor, 
                embedding_dim
            )
        else:
            self.downsample_layer = nn.AvgPool1d(
                kernel_size = downsample_factor, 
                stride = downsample_factor
            )
    
    def forward(self, x, mems):
        # Input is of shape T x B x C
        sf = self.downsample_factor

        if mems.numel() > 0:
            last_mems = mems[-1] # Outputs of the first stack of vanillas
            assert last_mems.size(0) > (sf - 1)
            x = torch.cat([last_mems[-(sf - 1):], x], dim = 0)
        else:
            padding = torch.zeros((sf - 1, x.size(1), x.size(2))).to(x.device)
            x = torch.cat([padding, x], dim = 0)

        if x.size(0) % sf > 0:
            # Hack to prevent destroing the tensor when T is divisible by sf
            x = x[:-(x.size(0) % sf)]

        assert x.size(0) % self.downsample_factor == 0, \
            'tgt_len not divisible by sf'

        # T x B x C
        if self.mode == 'linear':
            # T x B x C -> B x T x C
            x = x.transpose(0, 1)
            x = x.reshape(
                x.size(0), 
                x.size(1) // self.downsample_factor, 
                x.size(2) * self.downsample_factor
            )
            x = self.downsample_layer(x)
            x = x.transpose(0, 1)
        else:
            # T x B x C -> B x T x C -> B x C x T
            x = x.transpose(0, 1).transpose(1, 2)
            x = self.downsample_layer(x)
            x = x.transpose(1, 2).transpose(0, 1)

        return x 


class MemTransformerLM(nn.Module):
    def __init__(self, n_token, n_layer, n_head, d_model, d_head, d_inner,
                 dropout, dropatt, dtype, tie_weight=True, d_embed=None,
                 div_val=1, tie_projs=[False], pre_lnorm=False,
                 tgt_len=None, ext_len=None, mem_len=None,
                 cutoffs=[], adapt_inp=False,
                 same_length=False, attn_type=0, clamp_len=-1,
                 sample_softmax=-1,
                 funnel_config="[3, (1, 2) ,3]", 
                 funnel_resample='naive'):
        super(MemTransformerLM, self).__init__()
        self.n_token = n_token

        d_embed = d_model if d_embed is None else d_embed
        self.d_embed = d_embed
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head

        self.word_emb = AdaptiveEmbedding(n_token, d_embed, d_model, cutoffs,
                                          div_val=div_val)

        self.drop = nn.Dropout(dropout)

        self.tie_weight = tie_weight
        self.tie_projs = tie_projs
        self.div_val = div_val

        self.n_layer = n_layer

        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len
        self.max_klen = tgt_len + ext_len + mem_len
        
        # It is very important shit, we don't support that
        assert self.ext_len == 0

        self.attn_type = attn_type

        def create_decoder_layers(n_layers):
            layers = nn.ModuleList([
                RelPartialLearnableDecoderLayer(
                        n_head, d_model, d_head, d_inner, dropout,
                        tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len,
                        dropatt=dropatt, pre_lnorm=pre_lnorm)
                for _ in range(n_layers)
            ])

            return layers

        pre_layers, (funnel_layers, shorten_factor), post_layers = eval(funnel_config)
        assert funnel_resample in ['linear', 'naive'], \
                    'Now we only support two upsampling/downsampling methods'
        assert mem_len % shorten_factor == 0 and tgt_len % shorten_factor == 0, \
                    'Keep lengths divisible by sf'

        self.layers = nn.ModuleList([
            create_decoder_layers(pre_layers),
            Downsampler(
                embedding_dim=d_model,
                downsample_factor=shorten_factor,
                mode=funnel_resample
            ),
            create_decoder_layers(funnel_layers),
            Upsampler(
                embedding_dim=d_model,
                upsample_factor=shorten_factor,
                mode=funnel_resample
            ),
            create_decoder_layers(post_layers),
        ])

        self.sample_softmax = sample_softmax
        assert sample_softmax <= 0

        if tie_weight:
            emb_layers = [i.weight for i in self.word_emb.emb_layers]
        else:
            emb_layers = None

        emb_projs = self.word_emb.emb_projs

        self.crit = ProjectedAdaptiveLogSoftmax(n_token, d_embed, d_model,
                                                cutoffs, div_val=div_val,
                                                tie_projs=tie_projs,
                                                out_projs=emb_projs,
                                                out_layers_weights=emb_layers)

        self.same_length = same_length
        self.clamp_len = clamp_len

        self._create_params()

    def _create_params(self):
        self.pos_emb = PositionalEmbedding(self.d_model)
        self.r_w_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head).zero_())
        self.r_r_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head).zero_())

    def reset_length(self, tgt_len, ext_len, mem_len):
        if tgt_len < 1:
            raise RuntimeError(f'tgt_len should be >= 1, but got {tgt_len}')
        if ext_len < 0:
            raise RuntimeError(f'ext_len should be >= 0, but got {ext_len}')
        if mem_len < 0:
            raise RuntimeError(f'mem_len should be >= 0, but got {mem_len}')
        assert ext_len == 0
        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len

    def init_mems(self):
        if self.mem_len > 0:
            param = next(self.parameters())
            mems = []
            for i in range(len(self.layers)):
                layer = self.layers[i]
                if not isinstance(layer, Upsampler) and not isinstance(layer, Downsampler):
                    mems.append(
                        # We add + 1 here because we store hidden representations
                        # before first layer and after the last one
                        torch.empty(len(layer) + 1, 0, 
                                dtype=param.dtype, 
                                device=param.device)
                    )
            return mems
        else:
            return None

    def _update_mems(self, hids, mems, qlen, mlen, tgt_len, mem_len):
        if mems is None:
            return None

        assert len(hids) == len(mems), 'len(hids) != len(mems)'

        # There are `mlen + qlen` steps that can be cached into mems
        # For the next step, the last `ext_len` of the `qlen` tokens
        # will be used as the extended context. Hence, we only cache
        # the tokens from `mlen + qlen - self.ext_len - self.mem_len`
        # to `mlen + qlen - self.ext_len`.
        with torch.no_grad():
            stacked = torch.stack(hids)
            if (
                mem_len == tgt_len
                and stacked.size(1) == mem_len
            ):
                new_mems = stacked.detach()
            else:
                end_idx = mlen + qlen
                beg_idx = max(0, end_idx - mem_len)
                if mems.numel():
                    cat = torch.cat([mems, stacked], dim=1)
                else:
                    cat = stacked
                new_mems = cat[:, beg_idx:end_idx].detach()

        return new_mems
        

    def _forward(self, core_input, mems=None, layers = None, tgt_len = 0, mem_len = 0, clamp_len = 0):
        qlen, bsz, _ = core_input.size()        

        mlen = mems[0].size(0) if mems is not None else 0
        klen = mlen + qlen
        if self.same_length:
            all_ones = core_input.new_ones(qlen, klen)
            mask_len = klen - mem_len - 1
            if mask_len > 0:
                mask_shift_len = qlen - mask_len
            else:
                mask_shift_len = qlen
            dec_attn_mask = (torch.triu(all_ones, 1+mlen)
                             + torch.tril(all_ones, -mask_shift_len)).bool()
        else:
            dec_attn_mask = torch.triu(
                core_input.new_ones(qlen, klen), diagonal=1+mlen).bool()

        hids = []
        pos_seq = torch.arange(klen-1, -1, -1.0, device=core_input.device,
                                dtype=core_input.dtype)

        if clamp_len > 0:
            pos_seq.clamp_(max=clamp_len)

        pos_emb = self.pos_emb(pos_seq)
        pos_emb = self.drop(pos_emb)

        core_out = core_input

        for i, layer in enumerate(layers):
            hids.append(core_out.detach())
            mems_i = None if mems is None else mems[i]
            core_out = layer(core_out, pos_emb, self.r_w_bias,
                                self.r_r_bias, dec_attn_mask=dec_attn_mask,
                                mems=mems_i)

        hids.append(core_out.detach())
        new_mems = self._update_mems(hids, mems, qlen, mlen, tgt_len, mem_len)

        return core_out, new_mems


    def forward(self, data, target, mems):
        if mems is None:
            mems = self.init_mems() 

        tgt_len = target.size(0)

        # Token_ids to vector embeddings
        dec_inp = data
        word_emb = self.word_emb(dec_inp)
        hidden = self.drop(word_emb)

        # We don't do shift right because of the input/target structure
        # The target given to us is already shifted by one so we don't shift

        # Iterate over the model
        current_sf = 1
        mems_index = 0
        new_mems = []

        for i in range(len(self.layers)):
            layers = self.layers[i]

            if isinstance(layers, Upsampler):
                # The residual come from the latest downsampler
                # import pdb
                # pdb.set_trace()
                hidden = layers(hidden, new_mems[-2][-1][:tgt_len])
                current_sf = current_sf // layers.upsample_factor
            elif isinstance(layers, Downsampler):
                # input = 123, target = 234
                # 01 23

                # input = 456, target = 567
                # 34 56

                # input = 1234, target = 2345
                # 01 23 4

                # input = 567, target = 678
                # 45 67 8

                hidden = layers(hidden, mems[mems_index - 1])
                current_sf *= layers.downsample_factor
            else:
                hidden, new_mem = self._forward(
                    hidden, 
                    mems=mems[mems_index], 
                    layers=layers,
                    tgt_len=self.tgt_len // current_sf,
                    mem_len=self.mem_len // current_sf,
                    clamp_len=self.clamp_len // current_sf,
                )
                new_mems.append(new_mem)
                mems_index += 1


        hidden = self.drop(hidden)
        # Loss calculation
        loss = self.crit(hidden.view(-1, hidden.size(-1)), target.view(-1))
        loss = loss.view(tgt_len, -1)

        return (loss, new_mems)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='unit test')

    parser.add_argument('--n_layer', type=int, default=7, help='')
    parser.add_argument('--n_head', type=int, default=2, help='')
    parser.add_argument('--d_head', type=int, default=2, help='')
    parser.add_argument('--d_model', type=int, default=200, help='')
    parser.add_argument('--d_embed', type=int, default=200, help='')
    parser.add_argument('--d_inner', type=int, default=200, help='')
    parser.add_argument('--dropout', type=float, default=0.0, help='')
    parser.add_argument('--cuda', action='store_true', help='')
    parser.add_argument('--seed', type=int, default=1111, help='')
    parser.add_argument('--multi_gpu', action='store_true', help='')

    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda else "cpu")

    B = 3
    tgt_len, mem_len, ext_len = 2, 4, 0
    data_len = tgt_len * 20
    args.n_token = 10000

    import data_utils

    data = torch.LongTensor(data_len*B).random_(0, args.n_token).to(device)
    diter = data_utils.LMOrderedIterator(data, B, tgt_len, device=device, ext_len=ext_len)

    cutoffs = [args.n_token // 2]
    tie_projs = [False] + [True] * len(cutoffs)

    for div_val in [1, 2]:
        for d_embed in [200, 100]:
            model = MemTransformerLM(args.n_token, args.n_layer, args.n_head,
                                     args.d_model, args.d_head, args.d_inner,
                                     args.dropout, dropatt=args.dropout,
                                     tie_weight=True, d_embed=d_embed,
                                     div_val=div_val, tie_projs=tie_projs,
                                     pre_lnorm=True, tgt_len=tgt_len,
                                     ext_len=ext_len, mem_len=mem_len,
                                     cutoffs=cutoffs, attn_type=0,
                                     dtype=None, funnel_config="[4, (4, 2), 4]").to(device)

            mems = None
            for idx, (inp, tgt, seqlen, _) in enumerate(diter):
                _, mems = model(inp, tgt, mems)