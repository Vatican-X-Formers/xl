import os
import pdb
import torch
from tokenizers import Tokenizer
import sentencepiece as spm
from tokenizer import AutoregressiveTokeniser


class BoundaryCreator():
    def __init__(
        self, boundaries_type, boundary_ids=None,
        move_prob=0.0, deletion_prob=0.0, insert_prob=0.0,
        clamp_group_sizes=False, min_group_length=0, max_group_length=1000 * 1000,
        mean_normal=5.5, std_normal=1, fixed_sf=1, **kwargs,
    ):
        self.boundaries_type = boundaries_type

        if boundaries_type == 'ids':
            assert boundary_ids is not None and len(boundary_ids) > 0
        elif boundaries_type in ['constant', 'random_constant']:
            assert fixed_sf > 0
            self.fixed_sf = fixed_sf
        elif boundaries_type == 'normal':
            # Normal distribution arguments
            self.mean_normal = mean_normal
            self.std_normal = std_normal

        self.boundary_ids = boundary_ids

        # Corruption parameters
        self.move_prob = move_prob
        self.deletion_prob = deletion_prob
        self.insert_prob = insert_prob

        # Parameters to clamp group sizes
        self.clamp_group_sizes = clamp_group_sizes
        self.min_group_length = min_group_length
        self.max_group_length = max_group_length

    def corrupt_boundaries(self, boundaries):
        final_boundaries = torch.zeros_like(boundaries, dtype=torch.bool)

        if torch.rand(1).item() < self.move_prob:
            batch_ids, elems_ids = boundaries.nonzero(as_tuple=True)
            dir = torch.randint(low=0, high=6, size=(1,)).item()
            if dir < 3:
                dir -= 3
            else:
                dir = dir - 3 + 1
            moving_boundaries = ((elems_ids + dir) >= 0) & (boundaries.size(1) > (elems_ids + dir))
            batch_ids, elems_ids = batch_ids[moving_boundaries], elems_ids[moving_boundaries] + dir
            final_boundaries[(batch_ids, elems_ids)] = 1
        else:
            final_boundaries = boundaries

        batch_nonzero_ids, elems_nonzero_ids = final_boundaries.nonzero(as_tuple=True)
        batch_zero_ids, elems_zero_ids = (final_boundaries == 0).nonzero(as_tuple=True)

        # Delete
        non_zeros = batch_nonzero_ids.size(0)
        to_erase = int(non_zeros * self.deletion_prob)
        delete_boundaries = torch.randperm(non_zeros)[:to_erase].to(final_boundaries.device)
        final_boundaries[(batch_nonzero_ids[delete_boundaries], elems_nonzero_ids[delete_boundaries])] = 0

        # Insert
        zeros = batch_zero_ids.size(0)
        # Here I use non_zeros on purpose, I want insert and deletion prob to work similarly
        to_insert = int(non_zeros * self.insert_prob)
        insert_boundaries = torch.randperm(zeros)[:to_insert].to(final_boundaries.device)
        final_boundaries[(batch_zero_ids[insert_boundaries], elems_zero_ids[insert_boundaries])] = 1

        return final_boundaries

    def boundaries_from_group_sizes(self, boundaries, group_sizes):
        x = group_sizes.cumsum(dim=-1)
        y = (x < boundaries.size(1))
        batch_ids, seq_ids = y.nonzero(as_tuple=True)
        bound_ids = x[(batch_ids, seq_ids)]
        boundaries[(batch_ids, bound_ids)] = True
        return boundaries

    def restrict_max_group_length(self, boundaries):
        neg_x = (~boundaries)

        y = neg_x.cumsum(-1)

        y[neg_x] = 0

        y = y[:, 1:] - y[:, :-1].cummax(-1).values
        y = torch.cat([torch.zeros(boundaries.size(0), 1).to(boundaries.device), y], dim=-1)
        y[neg_x] = 0

        z = (neg_x.int() - y).cumsum(-1)

        return boundaries | ((z % self.max_group_length) == 0)

    def get_boundaries(self, txt=None, tensor=None):
        """
            Function that generates boundaries for given tensor of data

            Attributes:
                data - (torch.LongTensor) - [seq_len x batch_size]

            Returns:
                boundaries - (torch.BoolTensor) - [batch_size x seq_len]
        """
        assert tensor is not None
        data = tensor

        data = data.transpose(0, 1)  # batch_size x seq_len
        boundaries = torch.zeros_like(data, dtype=torch.bool)

        if self.boundaries_type == 'noboundaries':
            return None
        elif self.boundaries_type == "ids":
            for boundary_id in self.boundary_ids:
                boundaries |= (data == boundary_id)
        elif self.boundaries_type == "normal":
            group_sizes = torch.normal(mean=self.mean_normal,
                                       std=self.std_normal,
                                       size=data.size(),)
            group_sizes = group_sizes.round().clamp(self.min_group_length, self.max_group_length).long().to(data.device)
            boundaries = self.boundaries_from_group_sizes(boundaries, group_sizes)
        elif self.boundaries_type == "space_dist":
            # These are word lengths extracted from text8 dataset
            space_dist = [632308, 2401024, 3410951, 2733289, 2023812, 1447078, 1407995,
                          1071319, 765731, 517567, 282529, 162820, 87729, 38597, 13429]
            space_dist = torch.tensor(space_dist)

            group_sizes = torch.multinomial(input=space_dist.float(), num_samples=data.numel(), replacement=True)
            group_sizes = group_sizes.reshape(data.size()).long().to(data.device)
            group_sizes += 2  # This is the shift of the distribution
            boundaries = self.boundaries_from_group_sizes(boundaries, group_sizes)
        elif self.boundaries_type == 'constant':
            if self.fixed_sf == 1.5:
                boundaries[:, ::2] = 1
                boundaries[:, 1::4] = 1
            else:
                boundaries[:, ::self.fixed_sf] = 1
        elif self.boundaries_type == 'random_constant':
            for i in range(data.size(0)):
                how_much = data.size(1) // self.fixed_sf
                indexes = torch.randperm(data.size(1),
                                         device=data.device)[:how_much]
                boundaries[i, indexes] = 1
        else:
            raise NotImplementedError

        if self.move_prob > 0 or self.insert_prob > 0 or self.deletion_prob > 0:
            boundaries = self.corrupt_boundaries(boundaries)

        if self.clamp_group_sizes:
            boundaries = self.restrict_max_group_length(boundaries)

        return boundaries


class TokenizerBoundaryCreator(BoundaryCreator):
    def __init__(self, boundaries_type, tokenizer_type, tokenizer_vocab_size, tokenizer_dropout,
                 tokenizer_save_dir, tokenizer_algorithm, **kwargs):
        super().__init__(boundaries_type, **kwargs)

        self.tokenizer = AutoregressiveTokeniser(corpus_filepath='',
                                                 save_dir=tokenizer_save_dir,
                                                 tokenizer_type=tokenizer_type,
                                                 dropout=tokenizer_dropout,
                                                 algorithm=tokenizer_algorithm,
                                                 vocab_size=tokenizer_vocab_size)

    def get_boundaries(self, txt=None, tensor=None):
        """
            Function that generates boundaries for given tensor of data

            Attributes:
                data - (str) - just a string which we'd tokenize using tokenizer

            Returns:
                boundaries - (torch.BoolTensor) - [len(data)]
        """
        assert txt is not None
        data = txt
        return self.tokenizer.get_boundaries(data)


class NonAutoregressiveBoundaryCreator(BoundaryCreator):
    def __init__(self, boundaries_type, tokenizer_type, tokenizer_vocab_size, tokenizer_dropout,
                 tokenizer_save_dir, tokenizer_algorithm, **kwargs):
        super().__init__(boundaries_type, **kwargs)

        tokenizer_path = self.get_tokenizer_filename(tokenizer_type,
                                                     tokenizer_vocab_size)
        tokenizer_path = os.path.join(tokenizer_save_dir, 'json', tokenizer_path)
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

    def get_tokenizer_filename(self, tokenizer_type, vocab_size):
        filename = f'{tokenizer_type}-{vocab_size}.json'
        return filename

    def get_boundaries(self, txt=None, tensor=None):
        """
            Function that generates boundaries for given tensor of data
            Attributes:
                data - (str) - just a string which we'd tokenize using tokenizer
             Returns:
                 boundaries - (torch.BoolTensor) - [len(data)]
         """

        assert txt is not None
        data = txt
        boundaries = torch.zeros((len(data), len(data[0])), dtype=torch.bool)
        encoded_data = self.tokenizer.encode_batch(data)

        # HF tokenizer don't strip space in the end and tokenises it properly
        # No hacks needed
        for idx, x in enumerate(encoded_data):
            for a, _ in x.offsets:
                boundaries[idx, a] = True

        return boundaries


class SPMBoundaries(BoundaryCreator):
    def __init__(self, boundaries_type, tokenizer_type, tokenizer_vocab_size,
                 tokenizer_save_dir, **kwargs):
        super().__init__(boundaries_type, **kwargs)
        filename = self.get_tokenizer_filename(tokenizer_type,
                                               tokenizer_vocab_size)
        filepath = os.path.join(tokenizer_save_dir, 'spm', kwargs['dataset'], filename)
        filepath = 'spmunigram-5000.model'
        self.tokenizer = spm.SentencePieceProcessor(model_file=filepath)
        print(filepath)

    def get_tokenizer_filename(self, tokenizer_type, vocab_size):
        assert tokenizer_type.startswith('spm')
        filename = f'{tokenizer_type}-{vocab_size}.model'
        return filename

    def get_boundaries(self, txt=None, tensor=None):
        """
            Function that generates boundaries for given tensor of data
            In Unigram the boundary was always at the space position
            Also here I'm getting the segmentation, so I want to care that the
            model I use later either boundary starts or ends group is
            consistent with segmentation I extract here.

            # This segmentation has to be combined with boundary ends group

            Attributes:
                data - (torch.LongTensor) - [seq_len x batch_size]

            Returns:
                boundaries - (torch.BoolTensor) - [batch_size x seq_len]
        """
        assert txt is not None
        data = txt

        words_set = set()
        batch_size = len(data)

        for i in range(batch_size):
            words_set.update(data[i].split(' '))

        words_list = list(words_set)

        words_segmentation = {}

        for word, segmentation in zip(words_list,
                                      self.tokenizer.encode(words_list,
                                                            out_type=str)):
            if word == '':
                words_segmentation[''] = [0]
                continue
            else:
                assert len(segmentation)
                assert len(segmentation[0])
                assert segmentation[0].startswith('▁')

            if segmentation[0] == '▁':
                segmentation = segmentation[1:]
            else:
                segmentation[0] = segmentation[0][1:]

            words_segmentation[word] = [len(x) for x in segmentation]
            try:
                assert len(word) == sum(words_segmentation[word])
            except AssertionError:
                pdb.set_trace()

        sample_lengths = []

        for i in range(batch_size):
            words_lengths = [words_segmentation[word] for word in data[i].split(' ')]
            pieces_lengths = [((y + 1) if (i > 0 and j == (len(sublengths) - 1)) else y) for i, sublengths in enumerate(words_lengths) for j, y
                              in enumerate(sublengths)]
            sample_lengths.append(torch.tensor(pieces_lengths))

        total_lengths = [x.sum().item() for x in sample_lengths]
        assert len(set(total_lengths)) == 1
        assert total_lengths[0] == len(data[0])
        boundaries = torch.zeros(batch_size, total_lengths[0])

        for i in range(batch_size):
            boundaries[i, sample_lengths[i].cumsum(dim=0)[:-1]] = 1

        return boundaries


class MFBoundaries(BoundaryCreator):
    def __init__(self, boundaries_type, tokenizer_type, tokenizer_vocab_size,
                 tokenizer_save_dir, **kwargs):
        super().__init__(boundaries_type, **kwargs)
        filepath = os.path.join('data', 'mf', kwargs['dataset'], 'model.bin')
        import morfessor
        io = morfessor.MorfessorIO()
        self.tokenizer = io.read_binary_model_file(filepath)
        self.tokenizer.viterbi_segment('bovesin')

    def get_boundaries(self, txt=None, tensor=None, add_symbols=False, top_n=1):
        """
            Function that generates boundaries for given tensor of data
            In Unigram the boundary was always at the space position
            Also here I'm getting the segmentation, so I want to care that the
            model I use later either boundary starts or ends group is
            consistent with segmentation I extract here.

            # This segmentation has to be combined with boundary ends group

            Attributes:
                data - (torch.LongTensor) - [seq_len x batch_size]

            Returns:
                boundaries - (torch.BoolTensor) - [batch_size x seq_len]
        """
        assert txt is not None
        data = txt

        words_set = set()
        batch_size = len(data)

        for i in range(batch_size):
            words_set.update(data[i].split(' '))

        words_segmentation = {}

        for word in words_set:
            segmentation = self.tokenizer.viterbi_segment(word)[0]
            words_segmentation[word] = [len(x) for x in segmentation]

        words_segmentation[''] = [0]

        sample_lengths = []

        for i in range(batch_size):
            words_lengths = [words_segmentation[word] for word in data[i].split(' ')]
            pieces_lengths = [((y + 1) if (i > 0 and j == (len(sublengths) - 1)) else y) for i, sublengths in enumerate(words_lengths) for j, y
                              in enumerate(sublengths)]
            sample_lengths.append(torch.tensor(pieces_lengths))

        total_lengths = [x.sum().item() for x in sample_lengths]
        assert len(set(total_lengths)) == 1
        assert total_lengths[0] == len(data[0])
        boundaries = torch.zeros(batch_size, total_lengths[0])

        for i in range(batch_size):
            boundaries[i, sample_lengths[i].cumsum(dim=0)[:-1]] = 1

        return boundaries


def get_boundary_creator(boundaries_type, **kwargs):
    if boundaries_type in ['noboundaries', 'ids', 'normal', 'space_dist',
                           'constant', 'random_constant']:
        return BoundaryCreator(boundaries_type, **kwargs)
    elif boundaries_type == 'tokenizer':
        if kwargs['tokenizer_type'].startswith('spm'):
            # Sentencepiece approach was used as for second approach of fixing
            # tokenisers. It was used with boundary predictor that was trying
            # to learn the decisions made by tokenisers but autoregressively
            # I used the sentencepiece Unigram also for the 3rd approach of
            # fixing tokenisers
            return SPMBoundaries(boundaries_type, **kwargs)
        elif kwargs['tokenizer_type'].startswith('mf'):
            # Sentencepiece approach was used as for second approach of fixing
            # tokenisers. It was used with boundary predictor that was trying
            # to learn the decisions made by tokenisers but autoregressively
            # I used the sentencepiece Unigram also for the 3rd approach of
            # fixing tokenisers
            return MFBoundaries(boundaries_type, **kwargs)
        else:
            if kwargs['tokenizer_algorithm'] == 'approachna':
                # It corresponds to the second approach of fixing tokenisers
                return NonAutoregressiveBoundaryCreator(boundaries_type, **kwargs)
            else:
                # This branch corresponds to the first approach of fixing
                # tokenisers. It calculates the data from corpus, memorises all
                # all stats and makes autoregressive decisions based on it
                return TokenizerBoundaryCreator(boundaries_type, **kwargs)


if __name__ == '__main__':
    pass
