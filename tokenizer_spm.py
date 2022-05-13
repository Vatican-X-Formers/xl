import sentencepiece as spm
import sys
import os

assert len(sys.argv) >= 3
assert len(sys.argv) <= 4

tokens = int(sys.argv[1])
dataset = sys.argv[2]

if sys.argv[3] == 'split':
    split = True

prefix = os.path.join('data/', dataset, 'train.txt')
print(f'I take data from {prefix}')

tokenizer_name = f'spmunigram-{tokens}'

if split:
    tokenizer_name = 'split' + tokenizer_name

x = spm.SentencePieceTrainer.train(input=f'{prefix}',
                                   model_prefix=tokenizer_name,
                                   vocab_size=tokens,
                                   character_coverage=1.0,
                                   max_sentence_length=int(1e9),
                                   split_by_whitespace=True,
                                   split_digits=True,
                                   split_by_unicode_script=split,
                                   num_threads=32,
                                   max_sentencepiece_length=18,
                                   normalization_rule_name='identity',
                                   add_dummy_prefix=True,
                                   remove_extra_whitespaces=True,
                                  )
