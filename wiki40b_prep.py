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
specials = ['\u008A', '\u008B', '\u008C', '\u008D']
unk = '\u008E'


def transliteration_cleaners(text):
    '''Pipeline for non-English text that transliterates to ASCII.'''
    # text = unidecode.unidecode(text)
    text = only_whitelist(text)
    return text.strip()


def lowercase(text):
    return text.lower()


def collapse_whitespace(text):
    _whitespace_re = re.compile(r'\s+')
    return re.sub(_whitespace_re, ' ', text)


mapping = {}
base_path = 'data/wiki40b/'
threshold = 5

# for language in ['fi', 'de']:
for language in ['fi']:
    for split in ['train', 'validation', 'test']:
    # for split in ['test']:
        dataset = load_dataset('wiki40b', language, split=split, beam_runner='DirectRunner')

        # Concat articles
        text = ' '.join(dataset['text'])

        for sp in specials:
            assert sp not in text
        assert unk not in text

        for marker, special in zip(markers, specials):
            text = text.replace(marker, f' {special} ')

        text = lowercase(text)
        text = collapse_whitespace(text)
        text = text.strip()

        if split == 'train':
            counter = Counter(text)
            allowed_chars = [k for k, v in counter.items() if v > threshold]

        text = ''.join([c if c in allowed_chars else unk for c in text])

        with open(f'{base_path}/{language}/{split}.txt', 'w+') as file:
            file.write(text)
