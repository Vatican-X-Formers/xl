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
    general.add_argument('--algorithm', default=None)
    args, _ = parser.parse_known_args()

    args.corpus_filepath = os.path.join(args.corpus_dir, args.corpus_split)
    assert 0 <= args.dropout <= 1
    if args.dropout == 0:
        args.dropout = None
    return args


def split_text_to_chunks_on_spaces(corpus, chunks):
    corpus_len = len(corpus)
    chunk_len = (corpus_len + chunks - 1) // chunks
    split_indexes = [0]

    # split on spaces
    for i in range(chunk_len, corpus_len, chunk_len):
        for j in range(200):
            if corpus[i + j] == ' ':
                split_indexes.append(i + j)
                break

    split_indexes.append(corpus_len)

    return [corpus[left:right] for left, right in zip(split_indexes[:-1],
                                                      split_indexes[1:])]


def load_corpus_and_split(corpus_filepath, chunks):
    lines = []
    with open(corpus_filepath) as file:
        for i, line in enumerate(file):
            lines.append(line.strip())

    assert len(lines) == 1
    corpus = lines[0]
    return split_text_to_chunks_on_spaces(corpus, chunks)


class TokeniserTrainer():
    def __init__(self, corpus_filepath, save_dir, chunks=10000):
        self.corpus_filepath = corpus_filepath
        self.save_dir = save_dir
        self.chunks = chunks
        if corpus_filepath != '':
            self.train_data = load_corpus_and_split(corpus_filepath, chunks)
            assert len(self.train_data) == chunks

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
    def __init__(self, tokenizer, corpus_filepath, save_path, chunks=1000):
        self.tokenizer = tokenizer
        self.corpus_filepath = corpus_filepath
        self.save_path = save_path
        self.chunks = chunks
        self.train_data = load_corpus_and_split(corpus_filepath, chunks)

    def export_frequencies(self, corpus):
        words = corpus.strip().split(' ')
        words_counted = Counter(words)

        freqs = defaultdict(int)

        for word, occurences in words_counted.items():
            tokens = self.tokenizer.encode(word).tokens
            for token_id, token in enumerate(tokens):
                freqs[token] += occurences

                for i in range(1, len(token) + 1):
                    freqs[token[:i] + '*'] += occurences

                not_last = token_id != (len(tokens) - 1)
                if not_last:
                    # For approach 3 and 4
                    # We save the number of token's occurences but dependent on
                    # the starting character of the next token
                    freqs[token + '+' + tokens[token_id + 1][0]] += occurences

        return freqs

    def extract_data(self):
        print(f'Extracting data for {self.save_path}')
        global_freq = defaultdict(int)

        with Pool(multiprocessing.cpu_count()) as pool:
            for x in tqdm.tqdm(pool.imap_unordered(self.export_frequencies,
                                                   self.train_data)):
                for k, v in x.items():
                    global_freq[k] += v

        total_tokens = 0
        for k, v in global_freq.items():
            if not k.endswith('*'):
                total_tokens += v

        # Special key
        global_freq['+ALL+'] = total_tokens

        with open(self.save_path, 'wb') as file:
            pickle.dump(global_freq, file)

        os.chmod(self.save_path, 0o777)
        return global_freq


class AutoregressiveTokeniser():
    def __init__(self, corpus_filepath, save_dir, tokenizer_type, vocab_size,
                 dropout, algorithm):
        self.corpus_filepath = corpus_filepath
        self.save_dir = save_dir
        self.algorithm = algorithm

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
            tokenizer_data_extractor = TokenizersData(tokenizer,
                                                      corpus_filepath,
                                                      tokenizer_data_path)
            tokenizer_data = tokenizer_data_extractor.extract_data()

        self.tokenizer_data = tokenizer_data
        self.raw_tokenizer = Tokenizer.from_file(tokenizer_path)

    def approach1(self, current_subword, new_char):
        """
            We start a new group if there hasn't been a token in training
            corpora that starts with (current_subword + new_char)
        """
        if self.tokenizer_data[current_subword + new_char + '*'] == 0:
            return True
        else:
            return False

    def approach2(self, current_subword, new_char):
        """
            We start a new group if:
                p(current_subword + new_char + *) < p(current_subword) * p(new_char + *)
            That is, we compare probabilities of segmentation in which we
            either put a boundary or don't put a boundary. Probability of
            segmentation is here calculated as probabilities of tokens for
            unigram model trained on tokenized training corpora
        """
        return self.tokenizer_data[current_subword + new_char + '*'] * self.tokenizer_data['+ALL+'] < \
                self.tokenizer_data[current_subword] * self.tokenizer_data[new_char + '*']

    def approach2_5(self, current_subword, new_char):
        """
            We start a new group if:
                p(current_subword + new_char + *) < p(current_subword) * p(new_char + *)
            That is, we compare probabilities of segmentation in which we
            either put a boundary or don't put a boundary. Probability of
            segmentation is here calculated as probabilities of tokens for
            unigram model trained on tokenized training corpora
        """
        return self.tokenizer_data[current_subword + new_char + '*'] * self.tokenizer_data['+ALL+'] <= \
                self.tokenizer_data[current_subword] * self.tokenizer_data[new_char + '*']

    def approach3(self, current_subword, new_char):
        """
            Very similar to approach2, but we calculate the probabilities
            differently. Specifically here we ask explicitly about probability
            of starting a new group given a current_subword and a new_char.
            We put a boundary if negatives > positives
                positives - # of tokens that start with
                    current_subword+new_char
                negatives - # of tokens current_subword which are before the
                    token starting with the new_char
        """
        if self.tokenizer_data[current_subword + '+' + new_char] > \
                self.tokenizer_data[current_subword + new_char + '*']:
            return True
        else:
            return False

    def approach4(self, current_subword, new_char):
        """
           Combination of approach 2 and 3. We put a boundary if the
           probability from approach 3 that we finish current_subword
           multiplied by the probability of new_char* is greater than
           probability of token (current_subword+new_char*)
        """
        positives = self.tokenizer_data[current_subword + new_char + '*']
        negatives = self.tokenizer_data[current_subword + '+' + new_char]
        prob_of_finish = negatives / (positives + negatives)
        if prob_of_finish * self.tokenizer_data[new_char + '*'] > \
                self.tokenizer_data[current_subword + new_char + '*']:
            return True
        else:
            return False

    def get_boundary_predictor(self, algorithm):
        """
            All the approaches here are conditioned only on the
            single word, they don't use any larger context. For longer context
            there is the approach with additional module within the main
            network and the linear layer/transformer block.
            Here we could also add an approach that make use of tokenisation up
            until the current character, and not only the lastest subword.
        """
        return getattr(self, algorithm)

    def get_boundaries_for_word(self, word, algorithm=None):
        if algorithm is None:
            assert getattr(self, 'algorithm', None) is not None
            algorithm = self.algorithm
        boundary_predictor = self.get_boundary_predictor(algorithm)

        acc = '▁'
        boundaries = np.zeros(len(word))

        for i in range(len(word)):
            if boundary_predictor(acc, word[i]):
                boundaries[i] = True
                acc = word[i]
            else:
                acc += word[i]

        return boundaries

    def job(self, words, algorithm=None):
        words_dict = {}
        for word in words:
            boundaries = self.get_boundaries_for_word(word)
            words_dict[word] = boundaries
        return words_dict

    def job2(self, text):
        current_len = 0
        text_len = len(text)
        boundaries = np.zeros(text_len)
        words = text.split(' ')
        n_words = len(words)

        for idx, word in enumerate(words):
            if len(word) > 0:
                boundaries[current_len:current_len + len(word)] = self.words_boundaries[word]
                current_len += len(word)

            if not (idx + 1 == n_words):
                # space handling
                boundaries[current_len] = 1
                current_len += 1

        return boundaries

    def get_boundaries(self, text, algorithm=None, n_proc=16, chunks=100):
        words = text.split(' ')
        distinct_words = list(set(words))
        chunk_len = (len(distinct_words) + chunks - 1) // chunks
        jobs = [distinct_words[left:left + chunk_len] for left in range(0, len(distinct_words), chunk_len)]
        words_boundaries = {}

        print(f'There are {len(words)} in total to encode, \
              {len(distinct_words)} distinct ones, chunk_len = {chunk_len}')

        with Pool(n_proc) as pool:
            for job_dict in tqdm.tqdm(pool.imap_unordered(self.job, jobs), total=len(jobs)):
                for x, y in job_dict.items():
                    words_boundaries[x] = y

        self.words_boundaries = words_boundaries
        print('Finished encoding the words')

        jobs2 = split_text_to_chunks_on_spaces(text, chunks)
        boundaries_acc = []

        with Pool(n_proc) as pool:
            for boundaries in tqdm.tqdm(pool.imap(self.job2, jobs2), total=len(jobs)):
                boundaries_acc.append(boundaries)

        return torch.tensor(np.concatenate(boundaries_acc)).bool()


if __name__ == "__main__":
    args = parse_args()
    dataset = load_corpus_and_split(args.corpus_filepath, 1)[0][:500]
    auto_tokenizer = AutoregressiveTokeniser(args.corpus_filepath, args.save_dir,
                                        args.tokenizer_type, args.vocab_size,
                                        args.dropout, args.algorithm)
