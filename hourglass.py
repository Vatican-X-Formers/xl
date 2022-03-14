import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

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
    def __init__(self, d_model, d_inner, dropout, pre_lnorm=False, activation_function='relu'):
        super(PositionwiseFF, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        if activation_function == 'relu':
            activation_fn = nn.ReLU(inplace=True)
        elif activation_function == 'gelu':
            activation_fn = torch.nn.GELU()

        self.CoreNet = nn.Sequential(
            nn.Linear(d_model, d_inner), 
            activation_fn,
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

class RelPartialLearnableMultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                tgt_len=None, ext_len=None, mem_len=None, pre_lnorm=False,
                activation_function='None'):
        super(RelPartialLearnableMultiHeadAttn, self).__init__()

        del activation_function

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.q_net = nn.Linear(d_model, n_head * d_head)
        self.k_net = nn.Linear(d_model, n_head * d_head)
        self.v_net = nn.Linear(d_model, n_head * d_head)
        self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

    def _rel_shift(self, x):
        zero_pad = torch.zeros((x.size(0), x.size(1), x.size(2), 1),
                               device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=3)

        x_padded = x_padded.view(x.size(0), x.size(1), x.size(3) + 1, x.size(2))

        x = x_padded.narrow(2, 1, x_padded.size(2) - 1).view_as(x)

        return x

    def forward(self, w, r, r_w_bias, r_r_bias, attn_mask=None, mems=None):
        # w is of size: T x B x C
        # r is of size: T x 1 x C
        # biases are of size: (n_head x d_head), we add the same bias to each token
        # attn_mask is of size (q_len x k_len)
        # mems are of size: T x B x C
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)

        if mems is not None:
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                cat = self.layer_norm(cat)

            w_head_q, w_head_k, w_head_v = \
                map(lambda layer: layer(cat), [self.q_net, self.k_net, self.v_net])
            r_head_k = self.r_net(r)

            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_head_q, w_head_k, w_head_v = \
                    map(lambda layer: layer(self.layer_norm(w)), [self.q_net, self.k_net, self.v_net])
            else:
                w_head_q, w_head_k, w_head_v = \
                    map(lambda layer: layer(w), [self.q_net, self.k_net, self.v_net])
            r_head_k = self.r_net(r)

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
                                     pre_lnorm=kwargs.get('pre_lnorm'),
                                     activation_function=kwargs.get('activation_function')
                                     )

    def forward(self, dec_inp, r, r_w_bias, r_r_bias, dec_attn_mask=None, mems=None):
        output = self.dec_attn(dec_inp, r, r_w_bias, r_r_bias,
                               attn_mask=dec_attn_mask,
                               mems=mems)
        output = self.pos_ff(output)

        return output


class Upsampler(nn.Module):
    def __init__(self, embedding_dim, upsample_factor, mode='linear'):
        super().__init__()
        self.upsample_factor = upsample_factor
        self.mode = mode
        self.ln = nn.LayerNorm(embedding_dim)

    def forward(self, x, residual, upsampling_mask = None):
        if self.mode == 'naive':
            x = x.repeat_interleave(self.upsample_factor, dim=0)
        elif self.mode == 'custom':
            assert upsampling_mask is not None
            # T x B x C
            x = x.transpose(0, 1)
            x = torch.gather(
                x, 1, upsampling_mask.long().unsqueeze(-1).repeat(1, 1, x.size(-1))
            )
            x = x.transpose(0, 1)
            assert x.size() == residual.size()

        assert x.size(0) >= residual.size(0)
        # The upsampled vector can be longer than tgt_len, the len from just before shortening
        x = self.ln(x[:residual.size(0)]) + residual

        return x


class Downsampler(nn.Module):
    def __init__(self, embedding_dim, downsample_factor, mode='linear'):
        super().__init__()
        self.mode = mode
        self.downsample_factor = downsample_factor
        if mode == 'naive':
            self.downsample_layer = nn.AvgPool1d(
                kernel_size = downsample_factor,
                stride = downsample_factor
            )
        elif mode == 'custom':
            assert self.downsample_factor == 1, 'Just a special requirement of using custom mode'

        if mode in ['naive']:
            self.leftmost_group = nn.Parameter(
                torch.Tensor(self.downsample_factor - 1, 1, embedding_dim).zero_()
            )
        else:
            self.leftmost_group = nn.Parameter(torch.Tensor(1, 1, embedding_dim).zero_())

    def forward(self, x, mems = None, downsampling_mask = None):
        # Input is of shape T x B x C
        sf = self.downsample_factor

        if mems is not None and mems.numel() > 0:
            assert False, 'currently not supported'
            last_mems = mems[-1] # Outputs of the first stack of vanillas
            assert last_mems.size(0) > (sf - 1)
            x = torch.cat([last_mems[-(sf - 1):], x], dim = 0)
        else:
            if self.mode == 'linear' or self.mode == 'naive':
                x = torch.cat([self.leftmost_group.repeat(1, x.size(1) ,1), x], dim = 0)

        if x.size(0) % sf > 0:
            # Hack to prevent destroying the tensor when T is divisible by sf
            x = x[:-(x.size(0) % sf)]

        assert x.size(0) % sf == 0, \
            'tgt_len not divisible by sf'

        # T x B x C
        if self.mode == 'naive':
            # T x B x C -> B x T x C -> B x C x T
            x = x.transpose(0, 1).transpose(1, 2)
            x = self.downsample_layer(x)
            x = x.transpose(1, 2).transpose(0, 1)
        elif self.mode == 'custom':
            assert downsampling_mask is not None
            x = torch.einsum('tbc, bts -> sbc', x, downsampling_mask)
            x = torch.cat(
                [self.leftmost_group.repeat(1, x.size(1), 1), x], dim=0
            )

        return x

class BoundaryPredictor(nn.Module):
    def __init__(self, mode, d_model, threshold = 0.5):
        super().__init__()
        self.mode = mode
        self.threshold = threshold

        if mode == 'linear':
            self.boundary_predictor = nn.Linear(d_model, 1)
            self.loss = nn.BCEWithLogitsLoss(weight=torch.tensor([1, 1.5]).float())

    def forward(self, hidden, boundaries_gt=None):
        # Boundaries are of shape [seq_len x bs]
        # Hidden is of shape [seq_len x bs x d_model]
        if self.mode == 'linear':
            hidden = hidden[:-1]
            boundaries_gt = boundaries_gt[1:].float()

            preds = self.boundary_predictor(hidden).squeeze(-1)
            loss = self.loss(preds, boundaries_gt)

            preds = torch.sigmoid(preds) >= self.threshold

            boundaries_gt = boundaries_gt.bool()

            TP = ((preds == boundaries_gt) & preds).sum().item()
            FP = ((preds != boundaries_gt) & preds).sum().item()
            FN = ((preds != boundaries_gt) & (~preds)).sum().item()

            acc = (preds == boundaries_gt).sum().item() / boundaries_gt.numel()
            if TP == 0:
                precision, recall = 0, 0
            else:
                precision = TP/(TP+FP)
                recall = TP/(TP+FN)

            pad = torch.ones((1, preds.size(1)), dtype=preds.dtype, device=preds.device)
            preds = torch.cat([pad, preds], dim=0)

        return preds, loss, acc, precision, recall

class MemTransformerLM(nn.Module):
    def __init__(self, n_token, n_layer, n_head, d_model, d_head, d_inner,
                 dropout, dropatt, dtype, tie_weight=True, d_embed=None,
                 div_val=1, tie_projs=[False], pre_lnorm=False,
                 tgt_len=None, ext_len=None, mem_len=None,
                 cutoffs=[], adapt_inp=False,
                 same_length=False, attn_type=0, clamp_len=-1,
                 sample_softmax=-1,
                 funnel_config="[3, (1, 2) ,3]",
                 funnel_resample='naive',
                 activation_function='relu',
                 gather_stats=[],
                 boundary_predictor='none',
                 ):
        super(MemTransformerLM, self).__init__()
        self.n_token = n_token

        d_embed = d_model if d_embed is None else d_embed
        self.d_embed = d_embed
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head

        self.word_emb = nn.Embedding(n_token, d_model)

        self.drop = nn.Dropout(dropout)

        # Relative attention specific parameters
        self.pos_emb = PositionalEmbedding(self.d_model)
        self.r_w_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head).zero_())
        self.r_r_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head).zero_())

        self.tie_weight = tie_weight
        self.tie_projs = tie_projs
        self.div_val = div_val

        self.n_layer = n_layer

        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len

        self.pre_lnorm = pre_lnorm
        if self.pre_lnorm:
            self.layer_norms = nn.ModuleList([
                nn.LayerNorm(d_model) for _ in range(3)
            ])

        def create_decoder_layers(n_layers):
            layers = nn.ModuleList([
                RelPartialLearnableDecoderLayer(
                        n_head, d_model, d_head, d_inner, dropout,
                        tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len,
                        dropatt=dropatt, pre_lnorm=pre_lnorm,
                        activation_function=activation_function)
                for _ in range(n_layers)
            ])

            return layers

        pre_layers, (funnel_layers, shorten_factor), post_layers = eval(funnel_config)
        assert funnel_resample in ['linear', 'naive', 'custom'], \
                    'Now we only support two upsampling/downsampling methods'
        assert mem_len % shorten_factor == 0 and tgt_len % shorten_factor == 0, \
                    'Keep lengths divisible by sf'
        if post_layers == 0 and funnel_layers == 0 and shorten_factor == 1:
            print('You are not using funnel')
            self.layers = nn.ModuleList([
                create_decoder_layers(pre_layers)
            ])
        else:
            print(f'You are using funnel in config {funnel_config}')
            assert funnel_layers > 0
            assert post_layers == pre_layers, 'Our model is symmetric'
            assert shorten_factor > 1 or funnel_resample == 'custom'
            self.funnel_mode = funnel_resample
            self.layers = nn.ModuleList([
                create_decoder_layers(pre_layers),
                Downsampler(
                    embedding_dim=d_model,
                    downsample_factor=shorten_factor,
                    mode=funnel_resample,
                ),
                create_decoder_layers(funnel_layers),
                Upsampler(
                    embedding_dim=d_model,
                    upsample_factor=shorten_factor,
                    mode=funnel_resample
                ),
                create_decoder_layers(post_layers),
            ])
            self.boundary_predictor = BoundaryPredictor(mode=boundary_predictor, d_model=d_model)

        self.final_cast = nn.Linear(d_model, n_token)
        self.crit = torch.nn.CrossEntropyLoss(reduction='none')

        self.same_length = same_length
        self.clamp_len = clamp_len
        # Remember that this stats should be elementwise and not batch_agg
        # These stats, e.g. shortened_length, depend on the batch size
        # As we take maximum shortened_length from the batch - take care
        self.gather_stats = gather_stats

    def reset_length(self, tgt_len, ext_len, mem_len):
        if tgt_len < 1:
            raise RuntimeError(f'tgt_len should be >= 1, but got {tgt_len}')
        if ext_len < 0:
            raise RuntimeError(f'ext_len should be >= 0, but got {ext_len}')
        if mem_len < 0:
            raise RuntimeError(f'mem_len should be >= 0, but got {mem_len}')
        # assert ext_len == 0
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
                        torch.empty(len(layer), 0, 
                                dtype=param.dtype, 
                                device=param.device)
                    )
            return mems
        else:
            return None

    def _update_mems(self, hids, mems, qlen, mlen, mem_len):
        if mems is None:
            return None

        assert len(hids) == len(mems), 'len(hids) != len(mems)'

        with torch.no_grad():
            # The hids are of shape N_Layers x T x B x C
            stacked_hids = torch.stack(hids, dim = 0)
            if stacked_hids.size(1) == mem_len:
                new_mems = stacked_hids.detach()
            else:
                end_idx = mlen + qlen
                beg_idx = max(0, end_idx - mem_len)
                if mems.numel() > 0: # If mems contains any elements
                    concat_mems_hids = torch.cat([mems, stacked_hids], dim=1)
                else:
                    concat_mems_hids = stacked_hids
                new_mems = concat_mems_hids[:, beg_idx:end_idx].detach()

        return new_mems

    def _forward(self, core_input, mems=None, layers=None, mem_len=0, clamp_len=0):
        # Core_input is of size (T x B x C)
        # Mems is of size (N_layers x T x B x C)
        qlen, _, _ = core_input.size()
        mlen = mems[0].size(0) if mems is not None else 0
        klen = mlen + qlen

        if self.same_length:
            # By default we cheat a little bit as some tokens have more context
            # than the others. For example 100th token sees 99 tokens while 10th sees 9.
            # This branch, if enabled, ensures that all tokens has access to the same amount of tokens
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

        # After shift I get positive distance between token of interest and token to the left
        # We don't care about tokens to the right because we cannot look into future
        pos_seq = torch.arange(klen-1, -1, -1.0, device=core_input.device,
                                dtype=core_input.dtype)

        # This is used only during inference, as during inference we can use
        # longer sequences than during training as we don't have to store activations.
        if clamp_len > 0:
            pos_seq.clamp_(max=clamp_len)

        pos_emb = self.pos_emb(pos_seq)
        pos_emb = self.drop(pos_emb)

        core_out = core_input
        hids = []

        for i, layer in enumerate(layers):
            hids.append(core_out.detach())
            # Mems are none when mem_len is set to 0, we want to keep this possibility
            mems_i = None if mems is None else mems[i]
            core_out = layer(
                core_out, pos_emb, self.r_w_bias, self.r_r_bias,
                dec_attn_mask=dec_attn_mask, mems=mems_i
            )

        new_mems = self._update_mems(hids, mems, qlen, mlen, mem_len)

        return core_out, new_mems

    def create_masks(self, boundaries):
        # This is due to specific implementation we use to create pooling masks
        # later in the Transformer model - True assumes that we start a new group
        # and we always want to start a new group at the first element, not to skip anything
        boundaries = boundaries.transpose(0, 1)
        boundaries[:, 0] = True

        # Upsample mask creation
        upsample_mask = boundaries.cumsum(-1) - 1

        # Downsample mask creation
        n_segments = boundaries.sum(-1)
        maximum_segment = n_segments.max()
        tmp = torch.zeros_like(boundaries).unsqueeze(2) + torch.arange(1, maximum_segment + 1, 1, device = boundaries.device)
        foo = tmp - boundaries.cumsum(1).unsqueeze(-1)
        final = torch.zeros_like(foo)
        final[foo == 0] = 1
        size_of_groups = final.sum(1, keepdim=True)
        downsample_mask = final / (size_of_groups + 1e-9)

        return downsample_mask, upsample_mask, size_of_groups.squeeze(1).long().flatten()


    def forward(self, data, target, mems, boundaries=None, special=False):
        # I can keep track of training stats and log them
        stats = {}

        assert data.size(0) >= data.size(1)
        # Data and target are of size T x B
        if mems is None:
            mems = self.init_mems()

        # Data loader serves most batches of length args.tgt_len but
        # the last batch could be leftover and could be shorter
        # therefore we use actual length of a batch and not args.tgt_len
        # tgt_len = target.size(0)
        tgt_len = target.size(0) if target is not None else data.size(0)

        # Create masks if custom downsampling/upsampling
        if getattr(self, 'funnel_mode', None) == 'custom':
            downsampling_mask, upsampling_mask, size_of_groups = self.create_masks(boundaries)
            if 'shortened_length' in self.gather_stats:
                stats['shortened_length'] = downsampling_mask.size(2)
        else:
            downsampling_mask, upsampling_mask = None, None

        # Token_ids to vector embeddings
        word_emb = self.word_emb(data) # T x B x C
        hidden = self.drop(word_emb)

        # We don't do shift right because of the input/target structure
        # The target given to us is already shifted by one so we don't shift

        # Iterate over the model
        current_sf = 1 # We keep that to regulate mem_len and clamp_len in shortened layers
        mems_index = 0 # It points to current/next stack of Transformer layers for which we need mems
        new_mems = [] # Here we keep outputs and mems for the next steps, we also take residual from here
        loss_boundaries = 0 # default init beacause we later return it
        residual = None

        for i in range(len(self.layers)):
            layers = self.layers[i]

            if isinstance(layers, Upsampler):
                # The residual come from just before shortening
                # We take the last hids which are the final outputs of decoder stack before shortening
                # We also make sure to take last tgt_len elements as these are the actual outputs
                # residual_mems_id = mems_index - 2 # -1 are hids from funnel, -2 are from before shortening
                hidden = layers(hidden, residual=residual, upsampling_mask=upsampling_mask)
                current_sf = current_sf // layers.upsample_factor
            elif isinstance(layers, Downsampler):
                if special:
                    hidden = hidden.clone().detach()

                if getattr(self, 'boundary_predictor', None) is not None:
                    if self.boundary_predictor.mode == 'linear':
                        _, loss_boundaries, acc_boundaries, precision, recall = self.boundary_predictor(hidden, boundaries & (~(data == 0)))
                        stats['acc_boundaries'] = acc_boundaries
                        stats['loss_boundaries'] = loss_boundaries.item()
                        stats['precision'] = precision
                        stats['recall'] = recall

                if special:
                    return loss_boundaries, stats

                residual = hidden
                hidden = layers(hidden, mems[0] if mems is not None else None, downsampling_mask=downsampling_mask)
                current_sf *= layers.downsample_factor
            else:
                hidden, new_mem = self._forward(
                    hidden,
                    mems=mems[mems_index] if mems is not None else None,
                    layers=layers,
                    mem_len=self.mem_len // current_sf,
                    clamp_len=self.clamp_len // current_sf,
                )
                if self.pre_lnorm:
                    hidden = self.layer_norms[mems_index](hidden)
                new_mems.append(new_mem)
                mems_index += 1

        # Loss calculation, Negative log likelihood
        # What we do here is we calculate -log(softmax) over vocab
        # Then take the value corresponding only to our target

        hidden = hidden[-tgt_len:]
        if getattr(self, 'final_cast', None) is not None:
            logit = self.final_cast(hidden)
        else:
            logit = self.crit._compute_logit(hidden, self.crit.out_layers_weights[0], self.crit.out_layers_biases[0], self.crit.get_out_proj(0))

        if self.training or target is not None:
            # T x B x C
            assert hidden.size(0) == target.size(0)
            logit = logit.view(-1, logit.size(-1))
            target = target.view(-1)

            loss = self.crit(logit, target)
            loss = loss.view(tgt_len, -1)

            return loss, new_mems, stats, loss_boundaries
        else:
            # Generation mode, we return raw logits
            return logit


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
    tgt_len, mem_len, ext_len = 4, 12, 0
    data_len = tgt_len * 20
    args.n_token = 10000

    import data_utils

    data = torch.LongTensor(data_len*B).random_(0, args.n_token).to(device)
    diter = data_utils.LMOrderedIterator(data, B, tgt_len, device=device, ext_len=ext_len)

    cutoffs = [args.n_token // 2]
    tie_projs = [False] + [True] * len(cutoffs)

    div_val = 2
    d_embed = 100

    model = MemTransformerLM(args.n_token, args.n_layer, args.n_head,
                                args.d_model, args.d_head, args.d_inner,
                                args.dropout, dropatt=args.dropout,
                                tie_weight=True, d_embed=d_embed,
                                div_val=div_val, tie_projs=tie_projs,
                                pre_lnorm=True, tgt_len=tgt_len,
                                ext_len=ext_len, mem_len=mem_len,
                                cutoffs=cutoffs, attn_type=0,
                                same_length=True,
                                dtype=None, funnel_config="[6, (1, 2), 6]").to(device)

    mems = None
    for idx, (inp, tgt, seqlen, _) in enumerate(diter):
        _, mems = model(inp, tgt, mems)
  
