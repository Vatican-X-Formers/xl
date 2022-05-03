import torch
import pdb

char = torch.load('char.pt')
entropy = torch.load('entropy.pt')

print(f'Char is {char.mean()}')
print(f'Entropy is {entropy.mean()}')


def calc_entropy(logit):
    entropy = -torch.nn.functional.log_softmax(logit, dim=-1) * torch.nn.functional.softmax(logit, dim=-1)
    entropy = torch.sum(entropy, dim=-1)
    return entropy


def get_spikes(vector, threshold=0.5):
    vector = calc_entropy(vector)
    right = (vector[:-1, :] > vector[1:, :])
    left = (vector[1:, :] > vector[:-1, :])
    total = torch.cat([torch.ones((1, left.size(1)),
                                  device=left.device, dtype=left.dtype
                                  ), left])
    # total are better then their left
    total[:-1, :] &= right

    total = total & ~(vector <= threshold)

    return total


char_spikes, entropy_spikes = map(get_spikes, [char, entropy])

print(f'# of char spikes = {char_spikes.sum().item()}')
print(f'# of entropy spikes = {entropy_spikes.sum().item()}')

intersection = (char_spikes == entropy_spikes) & char_spikes

print(f'# intersection is {intersection.sum()}, which is \
      {intersection.sum().item() / entropy_spikes.sum().item()}')
