#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
from __future__ import print_function
from __future__ import division

def build_vocab_from(sentences):
    vocab_freq = dict()
    for x in sentences:
        for w in x:
            if w not in vocab_freq:
                vocab_freq[w] = 0
            vocab_freq[w] += 1
    vocab_freq = sorted(vocab_freq.iteritems(), key=lambda (w, c): c, reverse=True)
    vocab = dict()
    vocab['__PAD__'] = 0
    for v in vocab_freq:
        if v[1] > 5:
            vocab[v[0]] = len(vocab)
    return vocab

def load_vocab_from(file):
    vocab = dict()
    with open(file) as vf:
        for line in vf.readlines():
            sp_line = line.strip().decode('utf8').split('\t')
            vocab[sp_line[0]] = int(sp_line[1])
    return vocab

def merge_vocab(voc1, voc2):
    for v2, id2 in voc2.iteritems():
        if v2 not in voc1:
            voc1[v2] = len(voc1)
    return voc1

def to_word_ids(sentences, vocab):
    coded_sentences = list()
    for x in sentences:
        w_ids = list()
        for w in x:
            if w in vocab:
                w_ids.append(vocab[w])
        coded_sentences.append(w_ids)
    return coded_sentences
