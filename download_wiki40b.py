from datasets import load_dataset
from collections import Counter
import unidecode
import pdb
import re
import os
import sys

markers = ['\n_START_ARTICLE_\n', '\n_START_SECTION_\n', '\n_START_PARAGRAPH_\n', '_NEWLINE_']
specials = ['\u008A', '\u008B', '\u008C', '\u008D']
unk = '\u008E'

mapping = {}
base_path = 'data/wiki40b/'

language = sys.argv[1]
assert len(language)

os.makedirs(f'data/wiki40b/{language}', exist_ok=True)
os.chmod(f'data/wiki40b/{language}', 0o777)

for split in ['train', 'validation', 'test']:
    dataset = load_dataset('wiki40b', language, split=split, beam_runner='DirectRunner')

    assert unk not in dataset['text']
    text = unk.join(dataset['text'])

    os.makedirs(os.path.dirname(f'{base_path}{language}'), exist_ok=True)

    filename = f'{base_path}{language}/{split}.raw.txt'

    with open(filename, 'w+') as file:
        file.write(text)
