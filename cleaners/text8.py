"""Text cleaning script similar to wikifil.pl used to create text8.

Source: http://mattmahoney.net/dc/textdata.html#appendixa

Usage:

    ./convert.py INPUT_FILE [LANG]

"""
import re
import sys
from homoglyphs import transform_homoglyphs


alphabet = {
    'en': 'abcdefghijklmnopqrstuvwxyz\n',
}

numerals = {
    'en': {
        '0': 'zero',
        '1': 'one',
        '2': 'two',
        '3': 'three',
        '4': 'four',
        '5': 'five',
        '6': 'six',
        '7': 'seven',
        '8': 'eight',
        '9': 'nine',
    }
}

# wiki40b-specific markers
markers = ['\n_START_ARTICLE_\n', '\n_START_SECTION_\n',
           '\n_START_PARAGRAPH_\n', '_NEWLINE_']

lang = sys.argv[2] if len(sys.argv) > 2 else 'fi'

with open(sys.argv[1]) as f:
    text = f.read()

for m in markers:
    text = text.replace(m, ' ')

text = text.lower()

text = transform_homoglyphs(text)

for n, num in numerals[lang].items():
    text = text.replace(n, ' ' + num + ' ')

text = re.sub(f'[^{alphabet[lang]}]+', ' ', text).strip()
