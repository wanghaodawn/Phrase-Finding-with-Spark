#!/usr/bin/env python
# encoding: utf-8

"""
Generate word cloud.
"""

import sys
from wordcloud import WordCloud


def main():
    f_data = sys.argv[1]
    f_output = sys.argv[2]
    weights = {}
    with open(f_data) as f:
        for line in f:
            phrase, score = line.split(':')
            score = float(score)
            weights[phrase] = score

    wc = WordCloud().generate_from_frequencies(weights.items())
    wc.to_file(f_output)


if __name__ == '__main__':
    main()

