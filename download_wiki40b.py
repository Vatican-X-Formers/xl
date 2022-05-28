from datasets import load_dataset
import os
import sys

base_path = 'data/wiki40b/'

language = sys.argv[1]
assert len(language)

os.makedirs(f'data/wiki40b/{language}', exist_ok=True)
os.chmod(f'data/wiki40b/{language}', 0o777)

for split in ['train', 'validation', 'test']:
    dataset = load_dataset('wiki40b', language, split=split,
                           cache_dir='./cache', beam_runner='DirectRunner')

    text = '\n'.join(dataset['text'])

    os.makedirs(os.path.dirname(f'{base_path}{language}'), exist_ok=True)

    filename = f'{base_path}{language}/{split}.raw.txt'

    with open(filename, 'w+') as file:
        file.write(text)
