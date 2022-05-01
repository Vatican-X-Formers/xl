from datasets import load_dataset
from collections import Counter
import unidecode
import pdb
import re


"""
    The data is a list of articles, so I can easily concat them

    The output of this prep file is just a stream of data

    I can either do min frequency and replace less frequent chars with unk
    or do the unidecode on the whole corpora

    Default is using unidecode
"""


markers = ['\n_START_ARTICLE_\n', '\n_START_SECTION_\n', '\n_START_PARAGRAPH_\n', '_NEWLINE_']
specials = ['洙', '性', '金', '수']


def only_whitelist(text):
    # TODO
    # remember about chinese shit
    return text


def transliteration_cleaners(text):
    '''Pipeline for non-English text that transliterates to ASCII.'''
    text = unidecode.unidecode(text)
    for marker, special in zip(markers, specials):
        text = text.replace(marker, f' {special} ')
    text = lowercase(text)
    text = only_whitelist(text)
    text = collapse_whitespace(text)
    return text.strip()


def lowercase(text):
    return text.lower()


def collapse_whitespace(text):
    _whitespace_re = re.compile(r'\s+')
    return re.sub(_whitespace_re, ' ', text)


mapping = {}
base_path = 'data/wiki40b/'

# for language in ['fi', 'de']:
for language in ['fi']:
    for split in ['train', 'validation', 'test']:
    # for split in ['test']:
        dataset = load_dataset('wiki40b', language, split=split, beam_runner='DirectRunner')

        # Concat articles
        text = ' '.join(dataset['text'])

        # Remove repeated spaces
        text = transliteration_cleaners(text)

        counter = Counter(text)
        for idx, (k, v) in enumerate(counter.items()):
            mapping[k] = idx

        with open(f'{base_path}wiki40b_{language}_{split}.txt', 'w+') as file:
            # text = str(' '.join([str(mapping[c]) for c in text]))
            file.write(text)
