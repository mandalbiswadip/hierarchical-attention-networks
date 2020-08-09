import tensorflow as tf

try:
    from tensorflow.keras.layers import (
        Embedding,
        AdditiveAttention,
        Input, LSTM, RNN, SimpleRNN, Masking, Dense
    )
    from tensorflow.keras import Model
    from tensorflow.keras import backend as K
    from tensorflow import keras
except:
    from tensorflow.python.keras.layers import (
        Embedding,
        AdditiveAttention,
        Input, LSTM, RNN, SimpleRNN, Masking, Dense
    )
    from tensorflow.python.keras import Model
    from tensorflow.python.keras import backend as K
    from tensorflow.python import keras

from tensorflow.python.keras.layers.dense_attention import \
    _lower_triangular_mask, _merge_masks
from tensorflow.python.ops import nn
from tensorflow.python.ops import array_ops, math_ops
from .word_vector import word_list, word_vectors

ModelCheckpoint = keras.callbacks.ModelCheckpoint


# max number of sentences in a doc
max_sentences = 50

# max number of words in sentence
maxlen = 80
masking_value = 0


def get_attention_scores(self, inputs, mask=None):
    self._validate_call_args(inputs=inputs, mask=mask)
    q = inputs[0]
    v = inputs[1]
    k = inputs[2] if len(inputs) > 2 else v
    q_mask = mask[0] if mask else None
    v_mask = mask[1] if mask else None
    scores = self._calculate_scores(query=q, key=k)
    if v_mask is not None:
        # Mask of shape [batch_size, 1, Tv].
        v_mask = array_ops.expand_dims(v_mask, axis=-2)
    if self.causal:
        # Creates a lower triangular mask, so position i cannot attend to
        # positions j>i. This prevents the flow of information from the future
        # into the past.
        scores_shape = array_ops.shape(scores)
        # causal_mask_shape = [1, Tq, Tv].
        causal_mask_shape = array_ops.concat(
            [array_ops.ones_like(scores_shape[:-2]), scores_shape[-2:]],
            axis=0)
        causal_mask = _lower_triangular_mask(causal_mask_shape)
    else:
        causal_mask = None
    scores_mask = _merge_masks(v_mask, causal_mask)
    if scores_mask is not None:
      padding_mask = math_ops.logical_not(scores_mask)
      scores -= 1.e9 * math_ops.cast(padding_mask, dtype=K.floatx())
    attention_distribution = nn.softmax(scores)
    return attention_distribution


AdditiveAttention.get_attention_scores = get_attention_scores


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
        self.embedding_len = embedding_len
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

    def get_config(self):
        config = super(SequenceAttentionLayer, self).get_config()
        config['num_hidden'] = self.num_hidden
        config["embedding"] = self.if_embedding
        config["embedding_len"] = self.embedding_len
        return config

    def call(self, inputs):
        # TODO include mask as input and make sure gets flowing
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

    def __init__(self, word_num_hiden, sentence_num_hidden, **kwargs):
        """
        :param word_num_hiden:
        :param sentence_num_hidden:
        """
        super(HierarchicalAttentionLayer,
              self).__init__()
        self.word_num_hiden = word_num_hiden
        self.sentence_num_hidden = sentence_num_hidden
        self.sentence_layer = SequenceAttentionLayer(num_hidden=word_num_hiden, embedding=True)
        # TODO include mask for document lstm
        self.document_layer = SequenceAttentionLayer(num_hidden=sentence_num_hidden)


    def get_config(self):
        config = super(HierarchicalAttentionLayer, self).get_config()
        config['word_num_hiden'] = self.word_num_hiden
        config["sentence_num_hidden"] = self.sentence_num_hidden
        return config


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

    def get_config(self):
        config = super(HierarchicalLSTMLayer, self).get_config()
        config['word_num_hiden'] = self.word_num_hiden
        config["sentence_num_hidden"] = self.sentence_num_hidden
        config["embedding_len"] = self.embedding_len
        return config


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


def get_model(word_num_hiden=100, sentence_num_hidden=400, type="lstm" ):
    # (batch size, # sentence, # word)
    inp = Input(shape=(None,None))

    # (batch size, sentence num hidden)
    if type=="attention":
        h = HierarchicalAttentionLayer(
            word_num_hiden=70, sentence_num_hidden=100)(inp)
    elif type=="lstm":
        h = HierarchicalLSTMLayer(
            word_num_hiden=word_num_hiden, sentence_num_hidden=sentence_num_hidden)(inp)
    # (batch size, 2)
    out = Dense(2, activation="softmax")(h)
    model = Model(inp, out)
    return model