try:
    import tensorflow.keras as keras
except:
    import tensorflow.python.keras as keras

Input, Dense =  keras.layers.Input, keras.layers.Dense
Model =  keras.models.Model

import kerastuner as kt

RandomSearch = kt.tuners.RandomSearch

from .model import HierarchicalLSTMLayer
    # , HierarchicalAttentionLayer

def build_model(hp):
    # (batch size, # sentence, # word)
    inp = Input(shape=(None, None))

    # (batch size, sentence num hidden)

    # choices
    word_num_hiden = hp.Choice(
        "word_num_hiden", values=[50,100,150])
    sentence_num_hidden = hp.Choice(
        "sentence_num_hidden", values=[600, 1000, 2000, 4000])

    h = HierarchicalLSTMLayer(
        word_num_hiden=word_num_hiden,
        sentence_num_hidden=sentence_num_hidden)(inp)
    # h = HierarchicalAttentionLayer(
    #     word_num_hiden=word_num_hiden,
    #     sentence_num_hidden=sentence_num_hidden)(inp)

    # (batch size, 2)
    out = Dense(2, activation="softmax")(h)
    model = Model(inp, out)
    # model.compile(optimizer
    model.compile(
            optimizer="adam",
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
    return model

tuner = RandomSearch(
    build_model,
    objective=kt.Objective("val_accuracy", direction="max"),
    max_trials=5,
    executions_per_trial=3,
    # directory='data/',
    project_name='sentiment')