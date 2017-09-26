#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
from __future__ import print_function
from __future__ import division
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import argparse
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers.wrappers import Bidirectional
from keras.layers.embeddings import Embedding
from keras.callbacks import Callback
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from preprocess import load_vocab_from, to_word_ids
from sklearn.metrics import precision_recall_fscore_support

parser = argparse.ArgumentParser(description='train.py')
parser.add_argument('-data', required=True, \
                    help='Prefix to the *_train.txt file.')
opt = parser.parse_args()
pretrained_embeddings_file = 'bio_nlp_vec/PubMed-shuffle-win-2.txt'

# hyperparameters
epochs = 100
batch_size = 16
hidden_size = 200
rnn_hidden_size = 400
dropout_rate = 0.5
max_sequence_length = 100

# category label to int dictionary
categories = dict([
    ('N', 0),
    ('P', 1),
    ])
category_keys = [k[0] for k in sorted(categories.iteritems(), key=lambda (k, v): v)]
num_output_class = len(categories)

# embedding array
embedding_array = None

def load_pretrained_embeddings(vocab, emb_file):

    emb_array = 0.002 * np.random.random_sample((len(vocab), hidden_size)) - 0.001
    emb_array[0] = 0.
    seen_words = []
    print("Loading embedding file")
    with open(emb_file) as emf:
        for line in emf.readlines():
            line_split = line.split()
            if len(line_split) == 2: continue
            word = line_split[0]
            if word not in vocab: continue
            embeddings = [float(e) for e in line_split[1:]]
            emb_array[vocab[word]] = embeddings
            seen_words.append(word)
    for w, _ in vocab.iteritems():
        if w not in seen_words:
            print("{} not found in pretrained embeddings".format(w))
    return emb_array

def read_corpus(train_file, test_file, train_vocab):

    train_lines = [l.strip().split('\t') for l in open(train_file).readlines()]
    x_train = [l[1].split() for l in train_lines]
    y_train = [categories[l[0]] for l in train_lines]
    y_train = np.array(y_train, dtype=np.int16)
    test_lines = [l.strip().split('\t') for l in open(test_file).readlines()]
    x_test = [l[1].split() for l in test_lines]
    y_test = [categories[l[0]] for l in test_lines]
    y_test = np.array(y_test, dtype=np.int16)

    x_train_coded = to_word_ids(sentences=x_train, vocab=train_vocab)
    x_test_coded = to_word_ids(sentences=x_test, vocab=train_vocab)

    # pad sequence to same lengths
    x_train_coded = pad_sequences(x_train_coded, maxlen=max_sequence_length, padding='post', truncating='post')
    x_test_coded = pad_sequences(x_test_coded, maxlen=max_sequence_length, padding='post', truncating='post')

    y_train = to_categorical(y_train, num_classes=len(categories))
    y_test = to_categorical(y_test, num_classes=len(categories))

    x_train = np.array(x_train_coded, dtype=np.int16)
    x_test = np.array(x_test_coded, dtype=np.int16)

    return x_train, x_test, y_train, y_test, train_vocab


class FscoreLogCallback(Callback):
    def __init__(self, filename):
        super(FscoreLogCallback, self).__init__()
        self.logfile = open(filename, 'w')

    def on_epoch_end(self, batch, logs={}):
        predicted = np.argmax(self.model.predict(self.validation_data[0]), axis=1)
        answers = np.argmax(self.validation_data[1], axis=1)
        prec, reca, fscore, sup = precision_recall_fscore_support(answers, predicted, average='binary')
        msg = "Precision:{:2.2f}% Recall:{:2.2f}% Fscore:{:2.2f}%".format(prec*100, reca*100, fscore*100)
        print(msg)
        self.logfile.write(msg)
        self.logfile.write('\n')
        self.logfile.flush()

    def on_train_end(self, logs=None):
        self.logfile.close()

def main():

    def build_model():
        model = Sequential()
        model.add(Embedding(len(train_vocab), hidden_size, weights=[embedding_array],\
                            input_length=max_sequence_length))
        model.add(Dropout(dropout_rate))
        model.add(Bidirectional(LSTM(rnn_hidden_size)))
        model.add(Dropout(dropout_rate))
        model.add(Dense(2, activation='softmax'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())
        return model

    train_vocab = load_vocab_from(opt.data + '.vocab')
    embedding_array = load_pretrained_embeddings(train_vocab, pretrained_embeddings_file)
    for fold_id in range(10):
        tfsession = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5)))
        K.set_session(tfsession)
        train_file = 'corpus/{}_f{}_train.txt'.format(opt.data, fold_id)
        test_file = 'corpus/{}_f{}_test.txt'.format(opt.data, fold_id)
        log_file = '{}_f{}.log'.format(opt.data, fold_id)
        x_train, x_test, y_train, y_test, _ = read_corpus(train_file, test_file, train_vocab)
        fscore_cb = FscoreLogCallback(log_file)
        model = build_model()
        print("Fold {}".format(fold_id))
        model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, \
                  callbacks=[fscore_cb], verbose=2)
        predicted = np.argmax(model.predict(x_test), axis=1)
        y_test_to_label = np.argmax(y_test, axis=1)
        prec, reca, fscore, sup = precision_recall_fscore_support(y_test_to_label, predicted, average='binary')
        print("Final Precision:{:2.2f}% Recall:{:2.2f}% Fscore:{:2.2f}%".format(prec*100, reca*100, fscore*100))

if __name__ == "__main__":
    main()
