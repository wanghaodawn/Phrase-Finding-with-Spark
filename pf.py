from __future__ import print_function

#!/usr/bin/env python
# encoding: utf-8

"""
@brief Phrase finding with spark
@param fg_year The year taken as foreground
@param f_unigrams The file containing unigrams
@param f_bigrams The file containing bigrams
@param f_stopwords The file containing stop words
@param w_phrase Weight of phraseness
@param w_info Weight of informativeness
@param n_workers Number of workers
@param n_outputs Number of top bigrams in the output
"""

import sys
from pyspark import SparkConf, SparkContext
from operator import add
import math


def collect_stop_words(filename):
    f = open(filename, 'r')

    stop_words = {}
    while 1:
        line = str(f.readline())
        # If line is null
        if not line:
            break

        stop_words[line.strip()] = 1

    return stop_words


def mergeCxyBxy(a, b):
    if not a.startswith("Cxy"):
        temp = a
        a = b
        b = temp
    return a + "\t" + b


def mergeCxBx(a, b):
    if not a.startswith("Cx"):
        temp = a
        a = b
        b = temp
    return a + "\t" + b


def clean_unigram(line):
    words = line[1].split("\t")
    res = [0, 0, 0, 0, 0, 0]
    i = 0
    for word in words:
        if word.startswith("Cxy"):
            num = int(word[word.find("Cxy") + 3:])
            res[0] = num
        elif word.startswith("Bxy"):
            num = int(word[word.find("Bxy") + 3:])
            res[1] = num
        elif word.startswith("Cx"):
            num = int(word[word.find("Cx") + 2:])
            res[2] = num
        elif word.startswith("Bx"):
            num = int(word[word.find("Bx") + 2:])
            res[3] = num
        elif word.startswith("Cy"):
            num = int(word[word.find("Cy") + 2:])
            res[4] = num
        elif word.startswith("By"):
            num = int(word[word.find("By") + 2:])
            res[5] = num
        else:
            res[i] = num

        i += 1

    return tuple(res)


def split_line(line):
    key = line[0]
    items = line[1].split("\t")

    Cxy = 0.0
    Bxy = 0.0
    Cx  = 0.0
    Bx  = 0.0
    Cy  = 0.0
    By  = 0.0
    for item in items:
        if item.startswith("Cxy"):
            Cxy = float(item[3:])
        elif item.startswith("Bxy"):
            Bxy = float(item[3:])
        elif item.startswith("Cx"):
            Cx  = float(item[2:])
        elif item.startswith("Bx"):
            Bx  = float(item[2:])
        elif item.startswith("Cy"):
            Cy  = float(item[2:])
        elif item.startswith("By"):
            By  = float(item[2:])

    return (key, (Cxy, Bxy, Cx, Bx, Cy, By),)


def compute_score(line, unique_unigrams, unique_bigrams, total_bigrams_fg, total_bigrams_bg, total_unigrams_fg, total_unigrams_bg, w_phrase, w_info):
    key = line[0]
    Cxy = line[1][0]
    Bxy = line[1][1]
    Cx  = line[1][2]
    Bx  = line[1][3]
    Cy  = line[1][4]
    By  = line[1][5]

    first_word_fg  = (Cx + 1) / (unique_unigrams + total_unigrams_fg)
    second_word_fg = (Cy + 1) / (unique_unigrams + total_unigrams_fg)
    bigram_fg = (Cxy + 1) / (unique_bigrams + total_bigrams_fg)
    bigram_bg = (Bxy + 1) / (unique_bigrams + total_bigrams_bg)

    # Phrase
    phrase = bigram_fg * (math.log(bigram_fg) - math.log(first_word_fg * second_word_fg))

    # Info
    info = bigram_fg * (math.log(bigram_fg) - math.log(bigram_bg))

    # Final score
    score = w_phrase * phrase + w_info * info

    key = key.replace(" ", "-")
    
    return (key, score,)



def main(argv):
    # parse args
    fg_year = int(argv[1])
    f_unigrams = argv[2]
    f_bigrams = argv[3]
    f_stopwords = argv[4]
    w_info = float(argv[5])
    w_phrase = float(argv[6])
    n_workers = int(argv[7])
    n_outputs = int(argv[8])

    """ configure pyspark """
    conf = SparkConf().setMaster('local[{}]'.format(n_workers))  \
                      .setAppName(argv[0])
    sc = SparkContext(conf = conf)

    # TODO: start your code here
    # Save stop words in dictionary
    stop_words = collect_stop_words(f_stopwords)


    # Remove stop words
    init_unigram_word = sc.textFile(f_unigrams).filter(lambda line: \
        not stop_words.has_key(line.split("\t")[0]))

    init_bigram_word = sc.textFile(f_bigrams).filter(lambda line: \
        not stop_words.has_key(line.split("\t")[0].split(" ")[0]) and \
        not stop_words.has_key(line.split("\t")[0].split(" ")[1]))


    # Get foreground and background grams
    # x\tCx
    fg_unigram_word = init_unigram_word.filter(lambda line:\
        int(line.split("\t")[1]) == fg_year).map(lambda line:\
        (line.split("\t")[0], "Cx" + line.split("\t")[2])).reduceByKey(lambda a, b:\
        "Cx" + str(int(a[2:]) + int(b[2:])))
    # x\tBx
    bg_unigram_word = init_unigram_word.filter(lambda line:\
        int(line.split("\t")[1]) != fg_year).map(lambda line:\
        (line.split("\t")[0], "Bx" + line.split("\t")[2])).reduceByKey(lambda a, b:\
        "Bx" + str(int(a[2:]) + int(b[2:])))

    # x y, Cxy
    fg_bigram_word = init_bigram_word.filter(lambda line: 
        int(line.split("\t")[1]) == fg_year).map(lambda line:
        (line.split("\t")[0], "Cxy" + line.split("\t")[2])).reduceByKey(lambda a, b:\
        "Cxy" + str(int(a[3:]) + int(b[3:])))
    # x y, Bxy
    bg_bigram_word = init_bigram_word.filter(lambda line: 
        int(line.split("\t")[1]) != fg_year).map(lambda line:
        (line.split("\t")[0], "Bxy" + line.split("\t")[2])).reduceByKey(lambda a, b:\
        "Bxy" + str(int(a[3:]) + int(b[3:])))

    # x y, Cxy\tBxy
    bigram_word = fg_bigram_word.union(bg_bigram_word).map(lambda line:
        (line[0], line[1])).reduceByKey(lambda a, b: mergeCxyBxy(a, b))
    # x, Cx\tBx
    unigram_word = fg_unigram_word.union(bg_unigram_word).map(lambda line:
        (line[0], line[1])).reduceByKey(lambda a, b: mergeCxBx(a, b))


    # x, x y\tCxy\tBxy
    x_bigram_word = bigram_word.map(lambda line:\
        (line[0].split(" ")[0], line[0] + "\t" + line[1]))
    x_y_unigram = unigram_word.map(lambda line:\
        (line[0], line[1]))

    x_joinned_bigram = x_bigram_word.leftOuterJoin(x_y_unigram)

    # y, x y\tCxy\tBxyCxBx
    y_bigram_word = x_joinned_bigram.map(lambda line:\
        (line[1][0].split("\t")[0].split(" ")[1], line[1][0] + "\t" + line[1][1]))

    y_joinned_bigram = y_bigram_word.leftOuterJoin(x_y_unigram)

    # x y, Cxy\tBxy\tCx\tBx\tCy\tBy
    unigram_all = y_joinned_bigram.map(lambda line:\
        (line[1][0].split("\t")[0], line[1][0][(line[1][0].find("\t") + 1):] + "\t" + line[1][1].replace("Cx", "Cy").replace("Bx", "By"))).reduceByKey(lambda a, b: a + b)

    # (x y, (Cxy, Bxy, Cx, Bx, Cy, By))
    process_line = unigram_all.map(lambda line: split_line(line)).reduceByKey(lambda a, b: a + b)


    # Get count
    unique_unigrams = int(init_unigram_word.count())
    unique_bigrams  = int(init_bigram_word.count())

    total_bigrams_fg = int(fg_bigram_word.map(lambda line: \
        int(line[1][3:])).reduce(lambda a, b: a + b))
    total_bigrams_bg = int(bg_bigram_word.map(lambda line: \
        int(line[1][3:])).reduce(lambda a, b: a + b))

    total_unigrams_fg = int(fg_unigram_word.map(lambda line: \
        int(line[1][2:])).reduce(lambda a, b: a + b))
    total_unigrams_bg = int(bg_unigram_word.map(lambda line: \
        int(line[1][2:])).reduce(lambda a, b: a + b))


    # (x-y, score)
    compute_res = process_line.map(lambda line: 
        compute_score(line, unique_unigrams, unique_bigrams, total_bigrams_fg, total_bigrams_bg, total_unigrams_fg, total_unigrams_bg, w_phrase, w_info)).reduceByKey(lambda a,b: a + b)


    # Output
    print_list = compute_res.takeOrdered(n_outputs, key = lambda line: -line[1])
    for item in print_list:
        print(item[0] + ":" + str(item[1]))


    # Save result to file
    # fg_unigram_word.saveAsTextFile("hdfs://pnn.stoat.pdl.local.cmu.edu:8020/user/haow2/fg_unigram_word")
    # bg_unigram_word.saveAsTextFile("hdfs://pnn.stoat.pdl.local.cmu.edu:8020/user/haow2/bg_unigram_word")

    # fg_bigram_word.saveAsTextFile("hdfs://pnn.stoat.pdl.local.cmu.edu:8020/user/haow2/fg_bigram_word")
    # bg_bigram_word.saveAsTextFile("hdfs://pnn.stoat.pdl.local.cmu.edu:8020/user/haow2/bg_bigram_word")

    # bigram_word.saveAsTextFile("hdfs://pnn.stoat.pdl.local.cmu.edu:8020/user/haow2/bigram_word")
    # unigram_word.saveAsTextFile("hdfs://pnn.stoat.pdl.local.cmu.edu:8020/user/haow2/unigram_word")
    
    # x_joinned_bigram.saveAsTextFile("hdfs://pnn.stoat.pdl.local.cmu.edu:8020/user/haow2/x_joinned_bigram")
    # y_joinned_bigram.saveAsTextFile("hdfs://pnn.stoat.pdl.local.cmu.edu:8020/user/haow2/y_joinned_bigram")
    
    # unigram_all.saveAsTextFile("hdfs://pnn.stoat.pdl.local.cmu.edu:8020/user/haow2/unigram_all")
    # unigram.saveAsTextFile("hdfs://pnn.stoat.pdl.local.cmu.edu:8020/user/haow2/unigram")
    
    # process_line.saveAsTextFile("hdfs://pnn.stoat.pdl.local.cmu.edu:8020/user/haow2/process_line")
    # compute_res.saveAsTextFile("hdfs://pnn.stoat.pdl.local.cmu.edu:8020/user/haow2/compute_res")

    """ terminate """
    sc.stop()


if __name__ == '__main__':
    main(sys.argv)

