import re
from nltk import sent_tokenize
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
        texts = [self.clean_text(text) for text in texts]
        return super(CustomTokenizer, self).texts_to_sequences(texts=texts)

    def doc_to_sequences(self, docs, sent_tokenizer = None):
        if sent_tokenizer is None:
            sent_tokenizer = sent_tokenize
        indices = []
        if isinstance(docs, Iterable):
            for doc in docs:
                doc = self.tokenize_sentence(doc=doc, sent_tokenizer=sent_tokenizer)
                indices.append(self.texts_to_sequences(texts=doc))
        return indices

    def tokenize_sentence(self, doc, sent_tokenizer = sent_tokenize):
        return [sent for sent in sent_tokenizer(doc) if sent]

    @staticmethod
    def clean_text(text):
        return re.sub(r"<br /><br />", " ", text).strip()




