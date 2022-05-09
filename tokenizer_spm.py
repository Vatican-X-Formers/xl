import sentencepiece as spm
import sys
import os

assert len(sys.argv) == 3
tokens = int(sys.argv[1])
dataset = sys.argv[2]

prefix = os.path.join('data/', dataset, 'train.txt')
print(f'I take data from {prefix}')

x = spm.SentencePieceTrainer.train(input=f'{prefix}',
                                   model_prefix=f'spmunigram-{tokens}',
                                   vocab_size=tokens,
                                   character_coverage=1.0,
                                   max_sentence_length=int(1e9),
                                   split_by_whitespace=True,
                                   add_dummy_prefix=True,
                                   remove_extra_whitespaces=True,
                                  )
