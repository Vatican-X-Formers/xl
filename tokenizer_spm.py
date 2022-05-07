import sentencepiece as spm
import sys
# prefix = '/home/pnawrot/piotrek/datasets/text8/train.txt.raw'

prefix = 'data/wiki40b/fi/train.txt'
tokens = int(sys.argv[1])

x = spm.SentencePieceTrainer.train(input=f'{prefix}',
                                   model_prefix=f'spmunigram-{tokens}-wiki',
                                   vocab_size=tokens,
                                   character_coverage=1.0,
                                   max_sentence_length=int(1e9),
                                   split_by_whitespace=True,
                                   add_dummy_prefix=True,
                                   remove_extra_whitespaces=True,
                                  )
