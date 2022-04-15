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
            core_out = self.CoreNet(self.layer_norm(inp))
            output = core_out + inp
        else:
            core_out = self.CoreNet(inp)
            output = self.layer_norm(inp + core_out)

        return output


class RelPartialLearnableMultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0, pre_lnorm=False,
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

    def forward(self, w, r, r_w_bias, r_r_bias, attn_mask=None):
        # w is of size: T x B x C
        # r is of size: T x 1 x C
        # biases are of size: (n_head x d_head), we add the same bias to each token
        # attn_mask is of size (q_len x k_len)
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)

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

    def forward(self, dec_inp, r, r_w_bias, r_r_bias, dec_attn_mask=None):
        output = self.dec_attn(dec_inp, r, r_w_bias, r_r_bias,
                               attn_mask=dec_attn_mask)
        output = self.pos_ff(output)

        return output


class Upsampler(nn.Module):
    def __init__(self, embedding_dim, mode):
        super().__init__()
        self.mode = mode
        self.ln = nn.LayerNorm(embedding_dim)

        if self.mode == 'add_positionals':
            self.pos_emb = PositionalEmbedding(embedding_dim)
            self.pos_emb_cast = nn.Linear(embedding_dim, embedding_dim)

    def position_within_group(self, boundaries):
        neg_x = (~boundaries)
        neg_x[:, 0] = False

        y = neg_x.cumsum(-1)
        y[neg_x] = 0
        y = y[:, 1:] - y[:, :-1].cummax(-1).values
        y = torch.cat([torch.zeros(boundaries.size(0), 1).to(boundaries.device), y], dim=-1)
        y[neg_x] = 0

        z = (neg_x.int() - y).cumsum(-1)

        return z + 1

    def forward(self, x, residual, upsampling_mask, boundaries):
        if self.mode == 'average':
            # T x B x C
            x = x.transpose(0, 1)
            x = torch.gather(
                x, 1, upsampling_mask.long().unsqueeze(-1).repeat(1, 1, x.size(-1))
            )
            x = x.transpose(0, 1)
            assert x.size() == residual.size()
        elif self.mode == 'add_positionals':
            # T x B x C
            x = x.transpose(0, 1)
            x = torch.gather(
                x, 1, upsampling_mask.long().unsqueeze(-1).repeat(1, 1, x.size(-1))
            )
            x = x.transpose(0, 1)
            assert x.size() == residual.size()

            positions = self.position_within_group(boundaries)
            pos_emb = self.pos_emb(positions.flatten())
            pos_emb = pos_emb.reshape(x.size())
            pos_emb = self.pos_emb_cast(pos_emb)
            x += pos_emb
        elif self.mode == 'add_to_last':

            x = x.transpose(0, 1)
            x = torch.gather(
                x, 1, upsampling_mask.long().unsqueeze(-1).repeat(1, 1, x.size(-1))
            )
            x = x.transpose(0, 1)
            assert x.size() == residual.size()

        # The upsampled vector can be longer from just before shortening
        assert x.size(0) >= residual.size(0)
        x = self.ln(x[:residual.size(0)]) + residual

        return x


class Downsampler(nn.Module):
    def __init__(self, embedding_dim, mode):
        super().__init__()
        self.mode = mode
        if mode == 'average':
            # In fixed group size shortening we actually don't need to use that
            # In variable group size approach we might wanna predict boundaries
            # based on data. In such a situaiton we can add a group
            # representation to indeces which are >= max(indices_in_the_group).
            # Therefore probably most of the time to the first group we might
            # need to add special trained vector and not the one gathered from
            # data to prevent autoregressive property
            self.leftmost_group = nn.Parameter(torch.Tensor(1, 1, embedding_dim).zero_())
        elif mode == 'lstm':
            self.downsampler = nn.LSTM(input_size=embedding_dim,
                                       hidden_size=embedding_dim,
                                       num_layers=1,
                                       batch_first=False)
            self.leftmost_group = nn.Parameter(torch.Tensor(1, 1, embedding_dim).zero_())
        elif mode == 'gru':
            self.downsampler = nn.GRU(input_size=embedding_dim,
                                      hidden_size=embedding_dim,
                                      num_layers=1,
                                      batch_first=False)
            self.leftmost_group = nn.Parameter(torch.Tensor(1, 1, embedding_dim).zero_())
        elif mode == 'add_group_length_emb':
            self.group_size_emb = nn.Embedding(120, embedding_dim)
            self.leftmost_group = nn.Parameter(torch.Tensor(1, 1, embedding_dim).zero_())
            # self.pos_emb = PositionalEmbedding(embedding_dim)
            # self.pos_emb_cast = nn.Sequential(
            #     nn.Linear(embedding_dim, embedding_dim),
            #     nn.ReLU(),
            # )
        elif mode == 'weighted_mean':
            self.leftmost_group = nn.Parameter(torch.Tensor(1, 1, embedding_dim).zero_())

    def forward(self, x, downsampling_mask, size_of_groups):
        # Input is of shape T x B x C
        if self.mode == 'average':
            downsampled_data = torch.einsum('tbc, bts -> sbc', x, downsampling_mask)
        elif self.mode in ['lstm', 'gru']:
            batch_size, vanilla_length, max_groups_in_batch = downsampling_mask.size()
            max_group_length = size_of_groups.max().item()
            # if it works I can further optimize this by processing nonzero groups only
            # n_groups = (size_of_groups != 0).sum()

            tmp = torch.cat(
                [torch.zeros((batch_size, 1), device=size_of_groups.device,
                             dtype=size_of_groups.dtype), size_of_groups],
                dim=1).cumsum(-1)[:, :max_groups_in_batch].flatten()

            # z is (max_groups_in_batch*batch_size) x max_group_length
            # all (max_groups_in_batch*batch_size) are independent and z will store indexes of elements for each group
            z = torch.arange(max_group_length, device=size_of_groups.device)[None, :].repeat(max_groups_in_batch * batch_size, 1) + tmp.unsqueeze(-1)
            z[z >= vanilla_length] = 0
            z = z.flatten()

            batch_indexes = torch.arange(batch_size, device=size_of_groups.device).repeat_interleave(max_groups_in_batch * max_group_length)

            padded_data_for_rnn = x[z, batch_indexes]
            padded_data_for_rnn = padded_data_for_rnn.reshape(max_group_length,
                                                              max_groups_in_batch * batch_size, x.size(-1))
            rnn_output = self.downsampler(padded_data_for_rnn)[0]
            extract_indexes = (size_of_groups.flatten() - 1).clamp(min=0)
            extracted_data = rnn_output[extract_indexes, torch.arange(extract_indexes.size(0), device=size_of_groups.device).long()]
            downsampled_data = extracted_data.reshape(max_groups_in_batch, batch_size, -1)
        elif self.mode == 'add_group_length_emb':
            downsampled_data = torch.einsum('tbc, bts -> sbc', x, downsampling_mask)
            downsampled_data += self.group_size_emb(size_of_groups).transpose(0, 1)
            # pos_emb = self.pos_emb(size_of_groups.t().float().flatten())
            # pos_emb = pos_emb.reshape(downsampled_data.size())
            # pos_emb = self.pos_emb_cast(pos_emb)
            # downsampled_data = self.ln(downsampled_data) + pos_emb
        elif self.mode == 'weighted_mean':
            downsampling_mask[downsampling_mask == 0] = -1e9
            fn = torch.nn.functional.softmax
            downsampling_mask = fn(downsampling_mask * torch.linspace(0, 10 * downsampling_mask.size(1), downsampling_mask.size(1), device=downsampling_mask.device)[None, :, None], dim=1)
            downsampled_data = torch.einsum('tbc, bts -> sbc', x, downsampling_mask)

        downsampled_data = torch.cat(
            [self.leftmost_group.repeat(1, x.size(1), 1), downsampled_data], dim=0
        )

        return downsampled_data


class BoundaryPredictor(nn.Module):
    def __init__(self, mode, capacity, d_model, weight, d_inner=2048, dropout=0.0, threshold=0.5):
        super().__init__()
        self.mode = mode
        self.threshold = threshold
        self.weight = weight

        if capacity == 'linear':
            self.boundary_predictor = nn.Linear(d_model, 1)
        elif capacity == 'nonlinear':
            self.boundary_predictor = nn.Sequential(
                nn.Linear(d_model, d_inner),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(d_inner, 2),
            )

        if mode == 'default':
            self.loss = nn.BCEWithLogitsLoss(weight=torch.tensor([weight]).float())
        elif mode in ['equalize']:
            self.loss = nn.BCEWithLogitsLoss(reduction='none')
        elif mode == 'rl':
            pass

    def forward(self, hidden, boundaries_gt=None):
        # Boundaries are of shape [seq_len x bs]
        # Hidden is of shape [seq_len x bs x d_model]
        if self.mode in ['default']:
            preds = self.boundary_predictor(hidden).squeeze(-1)
            loss = self.loss(preds, boundaries_gt.float())

            preds = torch.sigmoid(preds) >= self.threshold
        elif self.mode in ['equalize']:
            preds = self.boundary_predictor(hidden).squeeze(-1)
            loss = self.loss(preds, boundaries_gt.float())
            preds = torch.sigmoid(preds) >= self.threshold

            positive_loss = loss[boundaries_gt].mean()
            negative_loss = loss[~boundaries_gt].mean()
            loss = negative_loss + positive_loss * self.weight
        elif self.mode == 'rl':
            logits = self.boundary_predictor(hidden)
            policy = torch.distributions.categorical.Categorical(logits=logits)
            preds = policy.sample().bool()
            loss = policy.log_prob(preds)
            bias = ((torch.exp(policy.logits)[:, :, 1].sum()) - (hidden.size(0) * hidden.size(1) * 0.25)) / (hidden.size(0) * hidden.size(1))
            bias = torch.abs(bias)

        TP = ((preds == boundaries_gt) & preds).sum().item()
        FP = ((preds != boundaries_gt) & preds).sum().item()
        FN = ((preds != boundaries_gt) & (~preds)).sum().item()

        acc = (preds == boundaries_gt).sum().item() / boundaries_gt.numel()
        if TP == 0:
            precision, recall = 0, 0
        else:
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)

        return preds, loss, bias, acc, precision, recall


class MemTransformerLM(nn.Module):
    def __init__(self, n_token, n_head, d_model, d_head, d_inner,
                 dropout, dropatt, pre_lnorm=False, same_length=False,
                 clamp_len=-1, funnel_config="[3, (1, ) ,3]",
                 downsample_mode='average', upsample_mode='average',
                 activation_function='relu',
                 gather_stats=[], bp_mode='none', bp_capacity='none',
                 bp_weight=0.0, bp_switch_step=0,
                 rl_loss_combine='none',
                 ):
        super(MemTransformerLM, self).__init__()
        self.rl_loss_combine = rl_loss_combine
        self.n_token = n_token

        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head

        self.word_emb = nn.Embedding(n_token, d_model)
        self.drop = nn.Dropout(dropout)

        # Relative attention specific parameters
        self.pos_emb = PositionalEmbedding(self.d_model)
        self.r_w_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head).zero_())
        self.r_r_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head).zero_())

        self.pre_lnorm = pre_lnorm
        if self.pre_lnorm:
            self.layer_norms = nn.ModuleList([
                nn.LayerNorm(d_model) for _ in range(3)
            ])

        def create_decoder_layers(n_layers):
            layers = nn.ModuleList([
                RelPartialLearnableDecoderLayer(
                    n_head, d_model, d_head, d_inner, dropout,
                    dropatt=dropatt, pre_lnorm=pre_lnorm,
                    activation_function=activation_function)
                for _ in range(n_layers)
            ])

            return layers

        pre_layers, (funnel_layers, ), post_layers = eval(funnel_config)
        if post_layers == 0 and funnel_layers == 0:
            self.layers = nn.ModuleList([
                create_decoder_layers(pre_layers)
            ])
        else:
            self.layers = nn.ModuleList([
                create_decoder_layers(pre_layers),
                Downsampler(
                    embedding_dim=d_model,
                    mode=downsample_mode,
                ),
                create_decoder_layers(funnel_layers),
                Upsampler(
                    embedding_dim=d_model,
                    mode=upsample_mode,
                ),
                create_decoder_layers(post_layers),
            ])

            # Boundary predictor
            if bp_mode != 'none':
                self.boundary_predictor = BoundaryPredictor(mode=bp_mode,
                                                            capacity=bp_capacity,
                                                            d_model=d_model,
                                                            weight=bp_weight)
                self.bp_switch_step = bp_switch_step

        self.final_cast = nn.Linear(d_model, n_token)
        self.crit = torch.nn.CrossEntropyLoss(reduction='none')

        self.same_length = same_length
        self.clamp_len = clamp_len

        # Remember that this stats should be elementwise and not batch_agg
        # These stats, e.g. shortened_length, depend on the batch size
        # As we take maximum shortened_length from the batch - take care
        self.gather_stats = gather_stats

    def _forward(self, core_input, layers=None):
        # Core_input is of size (T x B x C)
        qlen, _, _ = core_input.size()
        klen = qlen

        if self.same_length:
            # By default we cheat a little bit as some tokens have more context
            # than the others. For example 100th token sees 99 tokens while 10th sees 9.
            # This branch, if enabled, ensures that all tokens has access to the same amount of tokens
            all_ones = core_input.new_ones(qlen, klen)
            mask_len = klen - 1
            if mask_len > 0:
                mask_shift_len = qlen - mask_len
            else:
                mask_shift_len = qlen
            dec_attn_mask = (torch.triu(all_ones, 1) + torch.tril(all_ones, -mask_shift_len)).bool()
        else:
            dec_attn_mask = torch.triu(
                core_input.new_ones(qlen, klen), diagonal=1).bool()

        # After shift I get positive distance between token of interest and token to the left
        # We don't care about tokens to the right because we cannot look into future
        pos_seq = torch.arange(klen - 1, -1, -1.0, device=core_input.device, dtype=core_input.dtype)

        # This is used only during inference, as during inference we can use
        # longer sequences than during training as we don't have to store activations.
        if self.clamp_len > 0:
            pos_seq.clamp_(max=self.clamp_len)

        pos_emb = self.pos_emb(pos_seq)
        pos_emb = self.drop(pos_emb)

        core_out = core_input
        for i, layer in enumerate(layers):
            core_out = layer(
                core_out, pos_emb, self.r_w_bias, self.r_r_bias,
                dec_attn_mask=dec_attn_mask
            )

        return core_out

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
        tmp = torch.zeros_like(boundaries).unsqueeze(2) + torch.arange(1, maximum_segment + 1, 1, device=boundaries.device)
        foo = tmp - boundaries.cumsum(1).unsqueeze(-1)
        final = torch.zeros_like(foo)
        final[foo == 0] = 1
        size_of_groups = final.sum(1, keepdim=True)
        downsampling_mask = final / (size_of_groups + 1e-9)

        return downsampling_mask, upsample_mask, size_of_groups.long().squeeze(1)

    def forward(self, data, target, boundaries=None, step=0):
        stats = {}

        # Data and target are of size T x B
        assert data.size(0) >= data.size(1)

        # Data loader serves most batches of length args.tgt_len but
        # the last batch could be leftover and could be shorter
        # therefore we use actual length of a batch and not args.tgt_len
        # tgt_len = target.size(0)
        tgt_len = target.size(0) if target is not None else data.size(0)

        # Token_ids to vector embeddings
        # T x B x C
        word_emb = self.word_emb(data)
        hidden = self.drop(word_emb)

        mems_index = 0
        loss_boundaries = torch.tensor(0, dtype=data.dtype, device=data.device)
        upsampling_mask, residual = None, None

        for i in range(len(self.layers)):
            layers = self.layers[i]

            if isinstance(layers, Upsampler):
                # The residual come from just before shortening
                hidden = layers(hidden, residual=residual,
                                upsampling_mask=upsampling_mask,
                                boundaries=boundaries)
            elif isinstance(layers, Downsampler):
                if getattr(self, 'boundary_predictor', None) is not None:
                    boundaries_preds, loss_boundaries, bias, \
                        acc_boundaries, precision, recall = self.boundary_predictor(hidden, boundaries)
                    stats['acc_boundaries'] = acc_boundaries
                    stats['precision'] = precision
                    stats['recall'] = recall
                    if step >= self.bp_switch_step:
                        boundaries = boundaries_preds
                        downsampling_mask, upsampling_mask, size_of_groups = self.create_masks(boundaries)
                        if 'shortened_length' in self.gather_stats:
                            stats['shortened_length'] = downsampling_mask.size(2)

                residual = hidden
                hidden = layers(hidden, downsampling_mask=downsampling_mask,
                                size_of_groups=size_of_groups)
            else:
                hidden = self._forward(
                    hidden,
                    layers=layers,
                )

                if self.pre_lnorm:
                    hidden = self.layer_norms[mems_index](hidden)

                mems_index += 1

        hidden = hidden[-tgt_len:]
        logit = self.final_cast(hidden)

        if self.training or target is not None:
            # T x B x C
            assert hidden.size(0) == target.size(0)
            logit = logit.view(-1, logit.size(-1))
            target = target.view(-1)

            loss = self.crit(logit, target)
            loss = loss.view(tgt_len, -1)

            seq_loss = loss.mean(dim=0, keepdim=True)
            seq_loss = (seq_loss - seq_loss.mean()) / seq_loss.std()

            loss_boundaries = loss_boundaries[-tgt_len:]
            nll_boundaries = loss_boundaries.mean(dim=0, keepdim=True)

            if self.rl_loss_combine == '1':
                loss_boundaries = -nll_boundaries * seq_loss
                loss_boundaries = loss_boundaries.mean()

            loss_boundaries_with_bias = loss_boundaries + bias

            stats['loss_boundaries'] = loss_boundaries.item()
            stats['bias'] = bias.item()

            return loss, stats, loss_boundaries_with_bias
        else:
            # Generation mode, we return raw logits
            return logit
