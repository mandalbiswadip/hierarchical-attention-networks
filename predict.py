#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd

from models.text import CustomTokenizer
from models.model import word_list, max_sentences, maxlen
print("load")

MODEL_PATH = ""

# Note: also change the layer type (lstm vs attention) while loading
#=================================================================================
# For attention + lstm:

# model = tf.keras.models.load_model(
#     MODEL_PATH,
#     custom_objects={"HierarchicalAttentionLayer": HierarchicalAttentionLayer}
# )

# For lstm:

# model = tf.keras.models.load_model(
#     MODEL_PATH,
#     custom_objects={"HierarchicalAttentionLayer": HierarchicalLSTMLayer}
# )

# =================================================================================

tokenizer = CustomTokenizer(word_list=word_list)


data = pd.read_csv("tagged_data.csv")
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
a = np.zeros((len(inputs), max_sentences, maxlen))

for row, x in zip(a, inputs):
    row[:len(x)] = x[:max_sentences]

# Define Model

from models.model import HierarchicalAttentionLayer, HierarchicalLSTMLayer

# from models.tuner import tuner

# model.compile(
#     optimizer="adam",
#     loss="categorical_crossentropy",
#     metrics=["acc"]
# )
# y = pd.get_dummies(data.sentiment).values
y = data.sentiment.values

print("data shape {}".format(a.shape))
print("Beginning prediction.....")

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={"HierarchicalAttentionLayer": HierarchicalAttentionLayer}
)
# model = tf.keras.models.load_model(
#     MODEL_PATH,
#     custom_objects={"HierarchicalAttentionLayer": HierarchicalLSTMLayer}
# )

prediction = model.predict(a)
data["prediction"] = np.argmax(prediction)
data.to_csv("prediction.csv", index=False)
