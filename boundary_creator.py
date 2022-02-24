import os
import pdb
import torch
from tokenizers import Tokenizer

class BoundaryCreator():
    def __init__(
        self, boundaries_type, boundaries_tokens=None, boundary_ids=None,
        move_prob=0.0, deletion_prob=0.0, insert_prob=0.0,
        clamp_group_sizes = False, min_group_length = 0, max_group_length = 1000*1000, 
        mean_normal = 5.5, std_normal = 1,
    ):
        del boundaries_tokens
        self.boundaries_type = boundaries_type
        if boundaries_type == 'ids':
            assert boundary_ids is not None and len(boundary_ids) > 0
        
        if boundaries_type in ['ids']:
            self.extract_offline = False
        elif boundaries_type in ['normal']:
            self.extract_offline = False

        self.boundary_ids = boundary_ids

        # Corruption parameters
        self.move_prob = move_prob
        self.deletion_prob = deletion_prob
        self.insert_prob = insert_prob

        # Parameters to truncate group sizes
        self.clamp_group_sizes = clamp_group_sizes
        self.min_group_length = min_group_length
        self.max_group_length = max_group_length

        # Normal distribution arguments
        self.mean_normal = mean_normal
        self.std_normal = std_normal

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
        to_erase = int(non_zeros*self.deletion_prob)
        delete_boundaries = torch.randperm(non_zeros)[:to_erase].to(final_boundaries.device)
        final_boundaries[(batch_nonzero_ids[delete_boundaries], elems_nonzero_ids[delete_boundaries])] = 0

        # Insert
        zeros = batch_zero_ids.size(0)
        # Here I use non_zeros on purpose, I want insert and deletion prob to work similarly
        to_insert = int(non_zeros*self.insert_prob)
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

    def get_boundaries(self, data):
        """
            Function that generates boundaries for given tensor of data

            Attributes:
                data - (torch.LongTensor) - [seq_len x batch_size]

            Returns:
                boundaries - (torch.BoolTensor) - [batch_size x seq_len]
        """

        if self.boundaries_type == 'noboundaries':
            return None

        data = data.transpose(0, 1) # batch_size x seq_len
        boundaries = torch.zeros_like(data, dtype=torch.bool)

        if self.boundaries_type == "ids":
            for boundary_id in self.boundary_ids:
                boundaries |= (data == boundary_id)
        elif self.boundaries_type == "normal":
            group_sizes = torch.normal(mean=self.mean_normal, 
                             std=self.std_normal, 
                             size=data.size(),
                             )
            group_sizes = group_sizes.round().clamp(self.min_group_length, self.max_group_length).long().to(data.device)
            boundaries = self.boundaries_from_group_sizes(boundaries, group_sizes)
        elif self.boundaries_type == "space_dist":
            # These are word lengths extracted from text8 dataset
            space_dist = [632308, 2401024, 3410951, 2733289, 2023812, 1447078, 1407995, \
                           1071319, 765731, 517567, 282529, 162820, 87729, 38597, 13429]
            space_dist = torch.tensor(space_dist)
            
            group_sizes = torch.multinomial(input=space_dist.float(), num_samples=data.numel() ,replacement=True)
            group_sizes = group_sizes.reshape(data.size()).long().to(data.device)
            group_sizes += 2 # This is the shift of the distribution
            boundaries = self.boundaries_from_group_sizes(boundaries, group_sizes)
        else:
            raise NotImplemented

        boundaries = self.corrupt_boundaries(boundaries) 

        return boundaries


class TokenizerBoundaryCreator(BoundaryCreator):
    def __init__(self, boundaries_type, boundaries_tokens, **kwargs):
        super().__init__(boundaries_type, boundaries_tokens, **kwargs)

        self.extract_offline = True

        if boundaries_type == 'gpt2':
            self.tokenizer = Tokenizer.from_pretrained("gpt2")
        elif boundaries_type == 'bpe':
            self.tokenizer = Tokenizer.from_file(f'data/tokenizer/{boundaries_type}-{boundaries_tokens}.json')
        else:
            raise NotImplementedError

    def get_boundaries(self, data, n_chunks = 100):
        """
            Function that generates boundaries for given tensor of data

            Attributes:
                data - (str) - just a string which we'd tokenize using tokenizer
                n_chunks - (int) - tokenizer's output representation is very memory extensive
                    we can limit the RAM memory used by this boundary extractor
                    by processing the long text in chunks

            Returns:
                boundaries - (torch.BoolTensor) - [len(data)]
        """
        
        # Here we store the ids under which the next groups start
        groups_beg_ids = []

        line = data.replace(' ', '').replace('_', ' ')
        line_len = len(line)
        chunk_len = line_len // n_chunks
        boundaries = torch.zeros(len(line), dtype=torch.bool)

        #print(f'We extract boundaries from input of length {line_len}, chunk_len is {chunk_len}')
        for i in range(n_chunks):
            l = chunk_len * i
            if i + 1 == n_chunks:
                # If it's last chunk, that we set right boundary to the last element to capture all
                r = line_len
            else:
                r = chunk_len * (i + 1)

            groups_beg_ids += [a + l for a, _ in self.tokenizer.encode(line[l:r]).offsets]
            #print(f'We finished {i}-th chunk out of {n_chunks}. We encoded segment from {l} to {r}')

        assert r == line_len
        boundaries[groups_beg_ids] = True

        return boundaries


def get_boundary_checkpoint_name(datadir, boundaries_type, boundaries_tokens):
    if boundaries_type in ['noboundaries', 'ids', 'constant', 'normal', 'space_dist']:
        filename = os.path.join(datadir, 'cache.pt')
    elif boundaries_type in ['gpt2']:
        filename = os.path.join(datadir, f'cache_{boundaries_type}.pt')
    elif boundaries_type in ['bpe', 'wordpieces', 'sentencepieces']:
        assert boundaries_tokens > 0 
        filename = os.path.join(datadir, f'cache_{boundaries_type}_{boundaries_tokens}.pt')
    return filename


def get_boundary_creator(boundaries_type, **kwargs):
    if boundaries_type in ['noboundaries', 'ids', 'normal', 'space_dist', 'constant']:
        return BoundaryCreator(boundaries_type, **kwargs)
    elif boundaries_type in ['gpt2', 'bpe', 'sentencepieces', 'wordpieces']:
        return TokenizerBoundaryCreator(boundaries_type, **kwargs)
    else:
        return NotImplementedError


if __name__ == '__main__':
    pass
