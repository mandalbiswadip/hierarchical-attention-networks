try:
    from tensorflow.keras.preprocessing.text import Tokenizer
except:
    from tensorflow.python.keras.preprocessing.text import Tokenizer



from collections.abc import Iterable

class CustomTokenizer(Tokenizer):

    def __init__(self, word_list, oov_token="unk"):
        super(
            CustomTokenizer, self).__init__(oov_token=oov_token)
        self.word_list = word_list
        self.word_index =  {v:i for i,v in enumerate(self.word_list)}
        self.index_word = {i:v for i,v in enumerate(self.word_list)}

    def texts_to_sequences(self, texts):
        return super(CustomTokenizer, self).texts_to_sequences(texts=texts)

    def doc_to_sequences(self, docs, sent_tokenizer = "."):
        indices = []
        if isinstance(docs, Iterable):
            for doc in docs:
                doc = doc.split(sent_tokenizer)
                indices.append(self.texts_to_sequences(texts=doc))
        return indices



