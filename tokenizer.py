import os
import sys
import tqdm
import pickle
import argparse
import multiprocessing
from multiprocessing import Pool
from collections import Counter, defaultdict

import pdb
import torch
import numpy as np
from tokenizers import Tokenizer
from tokenizers.trainers import BpeTrainer, UnigramTrainer, WordPieceTrainer
from tokenizers.models import BPE, Unigram, WordPiece
from tokenizers.pre_tokenizers import Metaspace


def parse_args():
    parser = argparse.ArgumentParser()

    general = parser.add_argument_group('general setup')
    general.add_argument('--corpus_dir', default='/home/pnawrot/piotrek/datasets/text8')
    general.add_argument('--corpus_split', default='text8')
    general.add_argument('--save_dir', default='./tokenizer_data/')
    general.add_argument('--tokenizer_type', default='unigram')
    general.add_argument('--vocab_size', default=5000, type=int)
    general.add_argument('--dropout', default=0.0, type=float)
    args, _ = parser.parse_known_args()

    args.corpus_filepath = os.path.join(args.corpus_dir, args.corpus_split)
    assert 0 <= args.dropout <= 1
    if args.dropout == 0:
        args.dropout = None
    return args


class TokeniserTrainer():
    def __init__(self, corpus_filepath, save_dir, chunks=10000):
        self.corpus_filepath = corpus_filepath
        self.save_dir = save_dir

        lines = []
        with open(self.corpus_filepath) as file:
            for i, line in enumerate(file):
                lines.append(line.strip())

        assert len(lines) == 1
        corpus = lines[0]
        corpus_len = len(corpus)
        chunk_len = (corpus_len + chunks - 1) // chunks
        self.train_data = [corpus[left:left + chunk_len] for left in range(0, corpus_len, chunk_len)]

    def get_tokenizer_and_trainer(self, tokenizer_type, vocab_size, dropout):
        if tokenizer_type == 'bpe':
            return Tokenizer(BPE(dropout=dropout)), BpeTrainer(vocab_size=vocab_size)
        elif tokenizer_type == 'unigram':
            return Tokenizer(Unigram()), UnigramTrainer(vocab_size=vocab_size)
        elif tokenizer_type == 'wordpiece':
            return Tokenizer(WordPiece(max_input_chars_per_word=1000)), WordPieceTrainer(vocab_size=vocab_size)

    def get_tokenizer_filename(self, tokenizer_type, vocab_size, dropout):
        if dropout is not None and dropout > 0.0:
            return f'{tokenizer_type}-drop{dropout}-{vocab_size}.json'
        else:
            return f'{tokenizer_type}-{vocab_size}.json'

    def train(self, tokenizer_type, vocab_size, dropout):
        print(f'Setting up tokenizer training, Type:{tokenizer_type}, Vocab_size:{vocab_size}, Dropout: {dropout}')
        tokenizer, trainer = self.get_tokenizer_and_trainer(tokenizer_type,
                                                            vocab_size,
                                                            dropout)
        tokenizer.pre_tokenizer = Metaspace()
        tokenizer.train_from_iterator(self.train_data, trainer)

        filename = self.get_tokenizer_filename(tokenizer_type, vocab_size,
                                               dropout)
        checkpoint_path = os.path.join(self.save_dir, 'json', filename)
        print(f'Writing the tokeniser under: {checkpoint_path}')
        tokenizer.save(checkpoint_path)
        os.chmod(checkpoint_path, 0o777)

        print('Checkpoint verification')
        tokenizer_loaded = Tokenizer.from_file(checkpoint_path)
        print('Verification succesfull')
        return tokenizer_loaded


class TokenizersData():
    def __init__(self, tokenizer, corpus_filepath, save_dir, chunks=1000):
        self.tokenizer = tokenizer
        self.corpus_filepath = corpus_filepath
        self.save_dir = save_dir
        self.chunks = chunks
        with open(self.corpus_filepath) as file:
            train_data = []
            for i, line in enumerate(file):
                train_data.append(line.strip())
            assert len(train_data) == 1
            self.corpus = train_data[0]
        self.corpus_len = len(self.corpus)
        self.chunk_len = (self.corpus_len + self.chunks - 1) // self.chunks
        self.jobs = [(left, left + self.chunk_len) for left in range(0,
                                                                     self.corpus_len,
                                                                     self.chunk_len)]

    def export_frequencies(self, corpus):
        words = corpus.strip().split(' ')
        words_counted = Counter(words)

        freqs = defaultdict(int)

        for word, occurences in words_counted.items():
            tokens = self.tokenizer.encode(word).tokens
            for token in tokens:
                freqs[token] += occurences
                for i in range(1, len(token) + 1):
                    freqs[token[:i] + '*'] += occurences

        return freqs

    def extract_data(self):
        global_freq = defaultdict(int)

        with Pool(multiprocessing.cpu_count()) as pool:
            for x in tqdm.tqdm(pool.imap_unordered(self.export_frequencies,
                                                   self.jobs)):
                for k, v in x.items():
                    global_freq[k] += v

        total_tokens = 0
        for k, v in global_freq.items():
            if not k.endswith('*'):
                total_tokens += v

        self.log_probs = {}
        for k, v in global_freq.items():
            self.log_probs[k] = np.log(v) - np.log(self.total_tokens)

        with open(f'inference_{sys.argv[1].replace(".json", "")}.pkl', 'wb') as file:
            pickle.dump(global_freq, file)


class AutoregressiveTokeniser():
    def __init__(self, corpus_filepath, save_dir, tokenizer_type, vocab_size, dropout):
        self.corpus_filepath = corpus_filepath
        self.save_dir = save_dir

        trainer = TokeniserTrainer(self.corpus_filepath, self.save_dir)
        tokenizer_filename = trainer.get_tokenizer_filename(tokenizer_type,
                                                            vocab_size,
                                                            dropout)
        tokenizer_path = os.path.join(self.save_dir, 'json', tokenizer_filename)
        tokenizer_data_path = tokenizer_path.replace('json', 'pkl')

        if os.path.exists(tokenizer_data_path):
            print('Reading tokeniser\'s data from file')
            with open(tokenizer_data_path, 'rb') as file:
                tokenizer_data = pickle.load(file)
        else:
            if os.path.exists(tokenizer_path):
                print('Loading pretrained tokeniser')
                tokenizer = Tokenizer.from_file(tokenizer_path)
            else:
                print('Training a tokeniser from scratch')
                tokenizer = trainer.train(tokenizer_type, vocab_size, dropout)

            print('Extracting the necessary data from trained tokeniser')
            tokenizer_data_extractor = TokenizersData(tokenizer, corpus_filepath, save_dir)
            tokenizer_data = tokenizer_data_extractor.extract()

        self.tokenizer_data = tokenizer_data

    def approach1(self, word):
        return self.freq_total[word + '*'] == 0

    def approach2(self, word):
        if self.approach1(word):
            return True
        return self.freq_total[word + '*'] * self.total_tokens < self.freq_total[word[:-1]] * self.freq_total[word[-1] + '*']

    def get_boundary_predictor(self, algorithm):
        pass

    def get_boundaries_for_word(self, word, algorithm):
        boundary_predictor = self.get_boundary_predictor(algorithm)

        acc = '▁'
        boundaries = torch.zeros(len(word)).bool().cuda()

        for i in range(len(word)):
            acc += word[i]
            if boundary_predictor(acc):
                boundaries[i] = True
                acc = acc[-1]

        return boundaries.unsqueeze(1)

    def get_boundaries(self, text, algorithm):
        current_len = 0
        text_len = len(text)
        boundaries = torch.zeros(text_len).bool()

        for idx, word in enumerate(text.split(' ')):
            if len(word) > 0:
                boundaries[current_len:current_len + len(word)] = \
                    self.get_boundaries_for_word(word, algorithm)
                current_len += len(word)

            if not (idx + 1 == len(text.split(' '))):
                # space handling
                boundaries[current_len] = 1
                current_len += 1

        return boundaries


if __name__ == "__main__":
    args = parse_args()
    tokeniser = AutoregressiveTokeniser(args.corpus_filepath, args.save_dir,
                                        args.tokenizer_type, args.vocab_size,
                                        args.dropout)
    pdb.set_trace()
