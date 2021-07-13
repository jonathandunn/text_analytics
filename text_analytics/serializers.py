from gensim.models.keyedvectors import CompatVocab
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.corpora.dictionary import Dictionary
from gensim.test.utils import get_tmpfile
import scipy.sparse as sp
from helpers import NumpyEncoder
import numpy as np
import json
import pickle


class BaseSerializer:

    def __init__(self, new_object):
        try:
            self.obj = json.loads(new_object)
        except Exception as e:
            self.obj = new_object


class PhrasesSerializer(BaseSerializer):

    def serialize(self):
        return self.obj.export_phrases()

    def deserealize(self):
        return self.obj


class W2vEmbeddingSerializer(BaseSerializer):

    def serialize(self):
        return json.dumps(self.obj, cls=NumpyEncoder)

    def deserealize(self):
        return np.asarray(self.obj)


class W2vVocabSerializer(BaseSerializer):

    def serialize(self):
        res = {key: data.__dict__ for key, data in self.obj.items()}
        return json.dumps(res)

    def deseralize(self):
        res = {}
        for key, data in self.obj.items():
            compat = CompatVocab(**data)
            compat.count = data['count']
            res[key] = compat
        return res


class TfIdfSerializer(BaseSerializer):

    def get_tfidf_params(self):
        params = self.obj.get_params()
        params.pop('analyzer')
        params.pop('dtype')
        return params

    def serialize(self):
        payload = {"idf": self.obj.idf_.tolist(),
                   "vocabulary": self.obj.vocabulary_,
                   "params": self.get_tfidf_params()}
        return json.dumps(payload, cls=NumpyEncoder)

    def deserealize(self):
        idfs = np.asarray(self.obj['idf'])
        vectorizer = TfidfVectorizer(**self.obj['params'])
        # Monkey patch in order to indirectly fit a tfidf vectorizer.
        vectorizer._tfidf._idf_diag = sp.spdiags(idfs,
                                                 diags=0,
                                                 m=len(idfs),
                                                 n=len(idfs))
        vectorizer.vocabulary_ = self.obj['vocabulary']
        return vectorizer


class LdaDictionarySerializer(BaseSerializer):

    def serialize(self):
        temp_file = get_tmpfile('lda_dict_serialize_tmp')
        self.obj.save_as_text(temp_file)
        with open(temp_file, 'r') as saved_data:
            data = saved_data.read()
        return json.dumps({"corpus": data}, ensure_ascii=False)

    def deserealize(self):
        temp_file = get_tmpfile('lda_dict_deserialize_tmp')
        with open(temp_file, 'w') as te:
            te.write(self.obj['corpus'])
        return Dictionary.load_from_text(temp_file)


class LdaModelSerializer(BaseSerializer):

    def serialize(self):
        return pickle.dump(self.obj)

    def deserealize(self):
        return pickle.load(self.obj)


SERIALIZERS = {"phrases": PhrasesSerializer,
               "w2v_embedding": W2vEmbeddingSerializer,
               "w2v_vocab": W2vVocabSerializer,
               "tfidf_model": TfIdfSerializer,
               "lda_model": LdaModelSerializer}