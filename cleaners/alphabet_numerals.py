import re

alphabet = {
    'en': 'abcdefghijklmnopqrstuvwxyz\n',
    # https://en.wikipedia.org/wiki/Finnish_orthography
    'fi': 'abcdefghijklmnopqrstuvwxyzåäö\n'
},

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
    },
    'fi': {
        '0': 'nolla',
        '1': 'yksi',
        '2': 'kaksi',
        '3': 'kolme',
        '4': 'neljä',
        '5': 'viisi',
        '6': 'kuusi',
        '7': 'seitsemän',
        '8': 'kahdeksan',
        '9': 'yhdeksän',
    },
}


def spellout_digits(text, lang):
    for n, num in numerals[lang].items():
        text = text.replace(n, ' ' + num + ' ')

    return text


def keep_whitelist(text, lang):
    text = re.sub(f'[^{alphabet[lang]}]+', ' ', text).strip()
    return text
