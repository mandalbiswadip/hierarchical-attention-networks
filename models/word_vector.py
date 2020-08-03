import numpy as np

w = np.load("wordsList.npy")
word_vectors = np.load("wordVectors.npy")

decoder = np.vectorize(lambda x:x.decode("utf-8"))
word_list = decoder(w)
# 0th index is not useful
word_list[0] = "dummy_word"

# word_vector_dict = dict(zip(w, word_vectors))