"""Text cleaning script similar to wikifil.pl used to create text8.

Source: http://mattmahoney.net/dc/textdata.html#appendixa

Usage:

    ./convert.py INPUT_FILE [LANG]

"""
import re
import sys
from homoglyphs import transform_homoglyphs


