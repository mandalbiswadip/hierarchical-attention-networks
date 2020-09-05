#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd

from models.text import CustomTokenizer
from models.model import word_list, max_sentences, maxlen

tokenizer = CustomTokenizer(word_list = word_list)

batch_s = 16
split_len = 18000

data = pd.read_csv("tagged_data.csv")

data = data.filter(["text","sentiment"])
data = data.sample(frac=1)
# data = data.iloc[:10]
# =================================================
import tensorflow as tf
inp = tokenizer.doc_to_sequences(data.text.tolist())

inputs = []
for doc in inp:
    inputs.append(
        tf.keras.preprocessing.sequence.pad_sequences(
            doc, padding="post", value=0, maxlen=maxlen, dtype=None
        )
    )
a = np.zeros((len(inputs),max_sentences,maxlen))

for row,x in zip(a, inputs):
    row[:len(x)] = x[:max_sentences]


# Define Model

from models.model import get_model, ModelCheckpoint
from models.data import Sequence_generator
# from models.tuner import tuner

# model.compile(
#     optimizer="adam",
#     loss="categorical_crossentropy",
#     metrics=["acc"]
# )
# y = pd.get_dummies(data.sentiment).values
y = data.sentiment.values


x_train, y_train, x_test, y_test = a[:split_len], y[:split_len], a[split_len:], y[split_len:]

print("Train data shape {}".format(a.shape))
print("Beginning training.....")

for word_num_hiden in [ 150, 250]:
    for sentence_num_hidden in [200, 300, 400]:
        print("="*40)
        print("word hum hidden {}".format(word_num_hiden))
        print("sentence hum hidden {}".format(sentence_num_hidden))
        print("="*40)

        model = get_model(
            word_num_hiden=word_num_hiden,
            sentence_num_hidden=sentence_num_hidden,
            type="attention"
        )
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["acc"]
        )
        filepath = "model/attention-wh_{}_sh_{}-".format(
            word_num_hiden, sentence_num_hidden) + "{epoch:02d}-{val_acc:.2f}.h5"

        checkpoint = ModelCheckpoint(
            filepath, monitor='val_acc', verbose=1,
            save_best_only=True, mode='max'
        )

        model.fit(
                Sequence_generator(
                    x_train,y_train,
                    batch_s
                ),
                steps_per_epoch=int(len(x_train)/batch_s),
                epochs=30,
                verbose=2,
                validation_data=(x_test, y_test),
                callbacks=[checkpoint]
        )