try:
    from tensorflow.keras.utils import Sequence
except:
    from tensorflow.python.keras.utils import Sequence


import numpy as np

class Sequence_generator(Sequence):

    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
#         batch_x = keras.preprocessing.sequence.pad_sequences(batch_x,
#                                                              padding="post", value=-100)

        return batch_x, batch_y