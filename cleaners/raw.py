from collections import Counter
import sys
import os

from homoglyphs import normalise_homoglyphs
from alphabet_numerals import spellout_digits, keep_whitelist
from utils import wiki40b_markers, change_unknowns, collapse_whitespace


def text8_cleaner(text, lang):
    text = wiki40b_markers(text, mode='remove')
    text = text.lower()
    text = normalise_homoglyphs(text)
    text = spellout_digits(text, lang)
    text = keep_whitelist(text, lang)
    text = text.strip()
    return text


def soft_cleaner(text, threshold=5, valid_test_size=int(5e6)):
    text = wiki40b_markers(text, mode='keep')
    text = text.lower()

    # This cleaner has to be applied to concatenated train/valid/test
    # Apart from replacing least frequent symbols with \unk we also want
    # to change to unks all symbols that are not in train but in valid/test.
    import pdb
    pdb.set_trace()

    counter = Counter(text)
    allowed_chars = [k for k, v in counter.items() if v > threshold]

    text = collapse_whitespace(text)
    text = text.strip()


# Arguments
filename = sys.argv[1]
lang = sys.argv[2]
cleaner = sys.argv[3]

with open(filename) as file:
    text = file.read()

if cleaner == 'text8':
    text = text8_cleaner(text)
elif cleaner == 'soft':
    text = soft_cleaner(text)
else:
    raise NotImplementedError
