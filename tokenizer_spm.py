import sentencepiece as spm
import sys
import os

tokens = int(sys.argv[1])
dataset = sys.argv[2]

prefix = os.path.join('data/', dataset, 'train.txt')
print(f'I take data from {prefix}')

tokenizer_name = f'spmunigram-{tokens}'

tokenizer_name = os.path.join('tokenizer_data', 'spm', dataset, tokenizer_name)
print(f'I save the tokenizer at {tokenizer_name}')

x = spm.SentencePieceTrainer.train(input=f'{prefix}',
                                   model_prefix=tokenizer_name,
                                   vocab_size=tokens,
                                   character_coverage=1.0,
                                   max_sentence_length=int(1e9),
                                   split_by_whitespace=True,
                                   split_digits=True,
                                   split_by_unicode_script=True,
                                   num_threads=32,
                                   max_sentencepiece_length=18,
                                   normalization_rule_name='identity',
                                   add_dummy_prefix=True,
                                   remove_extra_whitespaces=True,
                                  )
