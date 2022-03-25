import sentencepiece as spm

prefix = '/home/pnawrot/piotrek/datasets/text8/'
tokens = 5000

x = spm.SentencePieceTrainer.train(input=f'{prefix}train.txt.raw,{prefix}valid.txt.raw',
                                   model_prefix=f'unigram-{tokens}',
                                   vocab_size=tokens,
                                   character_coverage=1.0,
                                   max_sentence_length=int(1e9),
                                   split_by_whitespace=True,
                                   add_dummy_prefix=True,
                                   remove_extra_whitespaces=True,
                                  )
