#!/usr/bin/env python
# coding: utf-8
"""
Create pretraining data
"""
import os

import pandas as pd

from models.text import CustomTokenizer
from models.model import word_list
print("load")

tokenizer = CustomTokenizer(word_list=word_list)
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
    {"text": neg_reviews, "sentiment": 0}
).append(pd.DataFrame(
    {"text": pos_reviews, "sentiment": 1}
))

print("Data Shape {}".format(data.shape))
print("Class Distribution {}".format(
    data.sentiment.value_counts())
)

data = data.reset_index()
data = data.filter(["text", "sentiment"])
data.to_csv("tagged_data.csv", index=False)