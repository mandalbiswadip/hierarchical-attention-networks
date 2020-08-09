#!/usr/bin/env python
# coding: utf-8

# In[2]:
import os

# In[49]:


import numpy as np
import pandas as pd


# In[53]:


from models.text import CustomTokenizer
from models.model import word_list, max_sentences, maxlen


tokenizer = CustomTokenizer(word_list = word_list)

batch_s = 32

path = "negativeReviews/"
neg_reviews = []
for f in os.listdir(path):
    file = os.path.join(path, f)
    with open(file, "r") as fl:
        neg_reviews.append(fl.read())
    

path = "positiveReviews//"
pos_reviews = []
for f in os.listdir(path):
    file = os.path.join(path, f)
    with open(file, "r") as fl:
        pos_reviews.append(fl.read())

data = pd.DataFrame(
    {"text":neg_reviews, "sentiment":0}
).append(pd.DataFrame(
    {"text":pos_reviews, "sentiment":1}
))

print("Data Shape {}".format(data.shape))
# data.to_csv("tagged_data.csv")
print("Class Distribution {}".format(
    data.sentiment.value_counts())
)

data = data.reset_index()

data = data.filter(["text","sentiment"])
data = data.sample(frac=1)
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
    row[:len(x)] = x[:50]


# Define Model

from models.model import get_model
from models.data import Sequence_generator
# from models.tuner import tuner

# model.compile(
#     optimizer="adam",
#     loss="categorical_crossentropy",
#     metrics=["acc"]
# )
# y = pd.get_dummies(data.sentiment).values
y = data.sentiment.values

x_train, y_train, x_test, y_test = a[:18000], y[:18000], a[18000:], y[18000:]

print("Beginning training.....")


for word_num_hiden in [150]:
    for sentence_num_hidden in [600, 1000, 2000, 4000]:
        print("word hum hidden {}".format(word_num_hiden))
        print("sentence hum hidden {}".format(sentence_num_hidden))

        model = get_model(
            word_num_hiden=word_num_hiden,
            sentence_num_hidden=sentence_num_hidden
        )
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["acc"]
        )
        model.fit(
                Sequence_generator(
                    x_train,y_train,
                    batch_s
                ),
                steps_per_epoch=int(len(x_train)/batch_s),
                epochs=300,
                verbose=2,
                validation_data=(x_test, y_test)
        )


# model.fit(
#         Sequence_generator(
#             x_train,y_train,
#             batch_s
#         ),
#         steps_per_epoch=int(len(x_train)/batch_s),
#         epochs=100,
#         verbose=2,
#         validation_data=(x_test, y_test)
# )

# tuner.search(
#         Sequence_generator(
#             x_train,y_train,
#             batch_s
#         ),
#         steps_per_epoch=int(len(x_train)/batch_s),
#         epochs=200,
#         verbose=2,
#         validation_data=Sequence_generator(
#             x_test, y_test, batch_s
#         ),
#         validation_steps = 1
# )