import tensorflow as tf

try:
    from tensorflow.keras.layers import (
        Embedding, AdditiveAttention,
        Input, LSTM, RNN, SimpleRNN, Masking, Dense
    )
    from tensorflow.keras import Model
except:
    from tensorflow.python.keras.layers import (
        Embedding, AdditiveAttention,
        Input, LSTM, RNN, SimpleRNN, Masking, Dense
    )
    from tensorflow.python.keras import Model


from .word_vector import word_list, word_vectors

# max number of sentences in a doc
max_sentences = 50

# max number of words in sentence
maxlen = 80
masking_value = 0


class SequenceAttentionLayer(tf.keras.layers.Layer):
    """Sequence model(LSTM) with attention sum of hidden states"""

    def __init__(
            self,
            num_hidden,
            embedding=False,
            embedding_len = 50
    ):
        """
        :param num_hidden:
        :param embedding:
        :param embedding_len:
        """
        super(
            SequenceAttentionLayer, self
        ).__init__()

        self.num_hidden = num_hidden
        self.if_embedding = embedding
        if embedding:
            self.embedding_len = embedding_len
            self.embed = Embedding(
                input_dim=len(word_vectors),
                output_dim=self.embedding_len,
                weights=[word_vectors],
                mask_zero=True,
                trainable=False
            )

        self.lstm = LSTM(
            self.num_hidden,
            activation="tanh",
            return_sequences=True
        )
        # pass the type of attention
        self.attention_layer = AdditiveAttention(
            use_scale=True
        )

    def call(self, inputs):
        # if embedding
        if self.if_embedding:
            inputs = self.embed(inputs)
            # putting every sentence in a single axis
            inputs = tf.reshape(
                inputs, shape = (-1 ,maxlen ,self.embedding_len)
            )

        lstm_out = self.lstm(inputs)
        h = self.attention_layer([lstm_out, lstm_out])
        out = tf.reduce_mean(h, axis=1)
        if self.if_embedding:
            # reshaping back to (batch_size, max seq len, embed length)
            out = tf.reshape(
                out,
                shape=(-1 ,max_sentences, self.num_hidden)
            )
        return out


class HierarchicalAttentionLayer(tf.keras.layers.Layer):
    """
    """

    def __init__(self, word_num_hiden, sentence_num_hidden):
        """
        :param word_num_hiden:
        :param sentence_num_hidden:
        """
        super(HierarchicalAttentionLayer,
              self).__init__()
        self.sentence_layer = SequenceAttentionLayer(num_hidden=word_num_hiden, embedding=True)
        self.document_layer = SequenceAttentionLayer(num_hidden=sentence_num_hidden)

    def call(self, input):
        sent_out = self.sentence_layer(
            input
        )
        doc_out = self.document_layer(sent_out)
        return doc_out

class HierarchicalLSTMLayer(tf.keras.layers.Layer):
    """
    """

    def __init__(self, word_num_hiden, sentence_num_hidden, embedding_len=50):
        """
        :param word_num_hiden:
        :param sentence_num_hidden:
        """
        super(HierarchicalLSTMLayer,
              self).__init__()


        self.word_num_hiden = word_num_hiden
        self.sentence_num_hidden = sentence_num_hidden

        self.embedding_len = embedding_len
        self.embed = Embedding(
            input_dim=len(word_vectors),
            output_dim=self.embedding_len,
            weights=[word_vectors],
            mask_zero=True,
            trainable=False
        )

        self.lstm_sent = LSTM(
            word_num_hiden,
            activation="tanh"
        )

        self.lstm_doc = LSTM(
            sentence_num_hidden,
            activation="tanh"
        )

    def call(self, inputs):
        # embed
        # reshape
        # sent lstm layer
        # reshape
        # document lstm layer
        inputs = self.embed(inputs)
        # putting every sentence in a single axis
        inputs = tf.reshape(
            inputs, shape = (-1 ,maxlen ,self.embedding_len)
        )
        sent_out = self.lstm_sent(
            inputs
        )
        # reshaping back to (batch_size, max seq len, embed length)
        sent_out = tf.reshape(
            sent_out,
            shape=(-1 ,max_sentences, self.word_num_hiden)
        )
        doc_out = self.lstm_doc(sent_out)
        return doc_out


# (batch size, # sentence, # word)
inp = Input(shape=(None,None))

# (batch size, sentence num hidden)
h = HierarchicalAttentionLayer(
    word_num_hiden=70, sentence_num_hidden=100)(inp)
# h = HierarchicalLSTMLayer(
#     word_num_hiden=70, sentence_num_hidden=100)(inp)
# (batch size, 2)
out = Dense(2, activation="softmax")(h)
model = Model(inp, out)
# model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["acc"])