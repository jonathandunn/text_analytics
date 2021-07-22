from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.phrases import Phrases
import pandas as pd
import pickle
import unittest

try:
    from serializers import W2vVocabSerializer, LdaDictionarySerializer, TfIdfSerializer
except:
    from .serializers import W2vVocabSerializer, LdaDictionarySerializer, TfIdfSerializer
    
try:
    from helpers import read_clean, clean
except:
    from .helpers import read_clean, clean

try:
    from settings import Settings
except:
    from .settings import Settings
    
#Initialize settings module
settings = Settings()

def read(df, column="Text"):
    return [str(x) for x in df.loc[:, column].values]

class SerializerTester(unittest.TestCase):

    def test_create_and_decode_phrases(self):
        df = pd.read_csv('text_analytics/tests/NYT.Corruption')
        phrases = Phrases(
            sentences=read_clean(df),
            min_count=100,
            threshold=0.70,
            scoring="npmi",
            max_vocab_size=100000000,
            delimiter="_",
        )
        exported = phrases.export_phrases()
        return exported

    def test_w2v(self):
        with open('text_analytics/tests/AI.State.Hotels.w2v.pickle', 'rb') as f:
            fo = pickle.load(f)

    def test_w2v_vocab(self):
        with open('text_analytics/tests/AI.State.NYT.w2v_vocab2.pickle', 'rb') as f:
            fo = pickle.load(f)
        data = W2vVocabSerializer(fo).serialize()
        data2 = W2vVocabSerializer(data).deseralize()

        
    def test_lda_dict(self):
        with open('text_analytics/tests/AI.State.Hotels.lda_dictionary3.pickle', 'rb') as f:
            fo = pickle.load(f)
        data = LdaDictionarySerializer(fo).serialize()
        data2 = LdaDictionarySerializer(data).deserealize()
        self.assertEqual(data2, fo)

    def test_fit_tfidf(self):
        from collections import defaultdict
        df = pd.read_csv('text_analytics/tests/NYT.Corruption')

        min_count = 100

        vocab = defaultdict(int)
        for line in read_clean(df):
            for word in line:
                vocab[word] += 1

        # Remove infrequent words and stopwords
        vocab_list = []

        # English-only version
        for word in vocab:
            if vocab[word] > min_count:
                if word not in FUNCTION_WORDS:
                    vocab_list.append(word)

        # Initialize TF-IDF Vectorizer
        vec = TfidfVectorizer(
            input="content",
            encoding="utf-8",
            decode_error="replace",
            ngram_range=(1, 1),
            analyzer=clean,
            vocabulary=vocab_list,
            norm="l2",
            use_idf=True,
            smooth_idf=True,
            tokenizer=None,
        )

        # Fit on the dataset
        print("Fitting TF-IDF")
        vec.fit(raw_documents=read(df))
        data = TfIdfSerializer(vec).serialize()
        data2 = TfIdfSerializer(data).deserealize()
        import pdb;pdb.set_trace()