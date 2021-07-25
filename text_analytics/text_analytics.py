#General imports
from collections import Mapping
from collections import defaultdict
from functools import partial
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.corpora import Dictionary
from gensim.models.phrases import Phrases, FrozenPhrases
from gensim.models import Word2Vec
from gensim.models.ldamodel import LdaModel
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report
from sklearn.metrics import adjusted_rand_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from scipy.spatial.distance import euclidean
from scipy.sparse import isspmatrix
from scipy.sparse import coo_matrix
from matplotlib import pyplot as plt
from wordcloud import WordCloud
from stop_words import safe_get_stop_words
from c2xg import C2xG
from corpus_similarity import Similarity
import spacy
import tensorflow as tf
import pandas as pd
import numpy as np
import cytoolz as ct
import pickle
import logging
import sys
import json
import codecs

#Package-internal imports
try:
    from helpers import clean, read_clean, clean_pre, clean_wordclouds, line_to_index, get_vocab, stream_clean
except:
    from .helpers import clean, read_clean, clean_pre, clean_wordclouds, line_to_index, get_vocab, stream_clean

try:
    from loader import ExternalFileLoader
except:
    from .loader import ExternalFileLoader

try:
    from settings import Settings
except:
    from .settings import Settings
    
try:
    from serializers import PhrasesSerializer, W2vEmbeddingSerializer, W2vVocabSerializer, TfIdfSerializer, LdaModelSerializer, LdaDictionarySerializer
except:
    from .serializers import PhrasesSerializer, W2vEmbeddingSerializer, W2vVocabSerializer, TfIdfSerializer, LdaModelSerializer, LdaDictionarySerializer
    
settings = Settings()

# TODO: Set logging for educational purposes.
ai_logger = logging.getLogger()
ai_logger.setLevel(logging.DEBUG)
stdout_handler = logging.StreamHandler(sys.stdout)
ai_logger.addHandler(stdout_handler)


class TextAnalytics:
    """Class to centralize the functions used in the course"""

    max_size = 200
    _style_vectorizer = None
    _sentiment_vectorizer = None
    _wordcloud = None
    _nlp = None
    lda = None
    lda_dictionary = None
    word_vectors = None
    word_vectors_vocab = None
    model = None
    classifier = None
    tfidf_vectorizer = None
    phrases = None

    def __init__(self, **kwargs):
        # Define word lists
        self.function_words_single = kwargs.get('function_words_single') \
            if kwargs.get('function_words_single') else settings.FUNCTION_WORDS_SINGLE
        self.function_words = kwargs.get('function_words') if kwargs.get('function_words') else settings.FUNCTION_WORDS
        self.positive_words = kwargs.get('positive_words') if kwargs.get('positive_words') else settings.POSITIVE_WORDS
        self.negative_words = kwargs.get('negative_words') if kwargs.get('negative_words') else settings.NEGATIVE_WORDS

        self.stop_words = self.function_words_single + self.positive_words + self.negative_words
        self.sentiment_words = self.positive_words + self.negative_words

        # Specific paths for the course labs
        self.data_dir = kwargs.get('data_dir') if kwargs.get('data_dir') else settings.DATA_DIR
        self.states_dir = kwargs.get('states_dir') if kwargs.get('states_dir') else settings.STATES_DIR

        self.loader = ExternalFileLoader(data_dir=self.data_dir, states_dir=self.states_dir)
        self.settings = Settings()
                
        self.serializers = {"phrases": PhrasesSerializer,
               "w2v_embedding": W2vEmbeddingSerializer,
               "w2v_vocab": W2vVocabSerializer,
               "tfidf_model": TfIdfSerializer,
               "lda_model": LdaModelSerializer,
               "lda_dictionary": LdaDictionarySerializer}
               
    def serialize(self, data, type, name):
    
        serializer = self.serializers[type]
        serialized = serializer(data).serialize()
        
        if type != "lda_model":
            with codecs.open(name, "w", encoding = "utf-8") as f:
                json.dump(serialized, f)
        
        elif type == "lda_model":
            with open(name, "wb") as f:
                pickle.dump(serialized, f)
        
        return
        
    def deserialize(self, type, name, language='en'):
    
        serializer = self.serializers[type]
        
        if type != "lda_model":
            with codecs.open(name, "r", encoding = "utf-8") as f:
                data = json.load(f)
         
        elif type == "lda_model":
            with open(name, "rb") as f:
                data = pickle.load(f)
            
        deserialized = serializer(data).deserialize()
        
        if type == "phrases":
            if language == 'en':
                common_terms = self.function_words_single
            else:
                common_terms = safe_get_stop_words(language)

            phrases = Phrases(delimiter="_", connector_words=common_terms)
            phrases.phrasegrams = deserialized
            deserialized = phrases        
        
        return deserialized
        
    def _get_vectorizer(self, ngrams, phraser=None, stop=None, nlp=None, vocab=None):
        return CountVectorizer(
            input="content",
            encoding="utf-8",
            decode_error="replace",
            ngram_range=ngrams,
            analyzer=partial(clean, phraser=phraser, stop=stop, nlp=nlp),
            vocabulary=vocab,
        )

    def load_data(self, filename):
        """
        Wrapper for loader get corpus method.
        :param filename:
        :return:
        """
        return self.loader.get_corpus(filename)

    def load_state(self, filename, file_type=None):
        """
        Wrapper for loader get state method
        :param filename:
        :param file_type:
        :return:
        """
        return self.loader.get_state(filename, state_type=file_type)

    @property
    def style_vectorizer(self):
        if not self._style_vectorizer:
            self._style_vectorizer = self._get_vectorizer(ngrams=(1, 2), vocab = self.function_words)
        return self._style_vectorizer

    @property
    def sentiment_vectorizer(self):
        if not self._sentiment_vectorizer:
            self._sentiment_vectorizer = self._get_vectorizer(ngrams=(1, 1), vocab = self.sentiment_words)
        return self._sentiment_vectorizer

    @property
    def wordcloud(self):
        if not self._wordcloud:
            self._wordcloud = WordCloud(width=1200,
                                        height=1200,
                                        max_font_size=75,
                                        min_font_size=10,
                                        max_words=200,
                                        background_color="white",
                                        relative_scaling=0.65,
                                        normalize_plurals=False,
                                        include_numbers=True,
                                        )
        return self._wordcloud

    def _get_vocab_list(self, df, n_features=None, min_count=1, language='en', return_freq=False, vocab=None, workers=1):
        """
        Gets vocab list
        :param df:
        :param min_count:
        :param language:
        :param return_freq:
        :return:
        """
        
        if not isinstance(vocab, Mapping):
            vocab = get_vocab(df, phraser=self.phrases, workers=workers)

        vocab_list = []
        
        if return_freq == True:
            return vocab

        if language == 'en':
            for word, freq in sorted(vocab.items(), key=lambda item: item[1], reverse=True):
               if freq > min_count:
                    if word not in self.function_words:
                        if word not in self.sentiment_words:
                            vocab_list.append(word)
        else:
            for word, freq in sorted(vocab.items(), key=lambda item: item[1], reverse=True):
                if freq > min_count:
                    vocab_list.append(word)
        
        if n_features is not None:
            vocab_list = vocab_list[:n_features]

        return vocab_list

    def _get_wordcloud_frequency_vocab(self, df, stage):
        """
        Gets wordclud frequency vocab TODO: Add better docs.
        :param df:
        :param stage:
        :return:
        """
        vocab = defaultdict(int)
        for line in self.read(df):
            line = clean_wordclouds(line, stage=stage, function_words_single=self.function_words_single)
            for word in line:
                vocab[word] += 1
        return vocab

    def _get_wordcloud_tfidf_vocab(self, df):
        """
        Gets wordcloud tfidf vocab
        :param df:
        :return:
        """
        x = self.tfidf_vectorizer.transform((" ".join(df.loc[:, "Text"].values),))
        # Make usable for the wordcloud package
        vocab = x.todense()
        columns = [k for k, v in sorted(self.tfidf_vectorizer.vocabulary_.items(), key=lambda item: item[1])]
        vocab = pd.DataFrame(vocab, columns=columns).T
        vocab = vocab.to_dict()
        return vocab[0]

    def _plot_wordcloud(self, stage, name):
        """
        Plots wordcloud
        :param stage:
        :param name:
        :return:
        """
        plt.figure()
        plt.imshow(self.wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=0.95, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.title(label="Cleaning: " + str(stage), fontdict=None, loc="center", pad=None)

        # Save to disk if a filename is given, otherwise just display it
        if name is not None:
            plt.savefig(name, format="tif", dpi=300, pad_inches=0)
            plt.close("all")
        else:
            plt.show()

    @staticmethod
    def read(df, column="Text"):
        """
        Returns a list of raw strings from the dataframe
        :param df:
        :param column:
        :return:
        """
        return [str(x) for x in df.loc[:, column].values]

    @staticmethod
    def save(name, file_contents):
        """
        Save current state: includes all features and classifiers
        :param name:
        :param file_contents:
        :return:
        """
        with open("AI.State." + name + ".pickle", "wb") as f:
            pickle.dump(file_contents, f)
            
    def get_association(self, df, min_count=1, threshold=0.70, save_phraser=False, language='en'):
    
        cxg = C2xG(language = self.settings.MAP_THREE[language])
        association_df = cxg.get_association(self.read(df), freq_threshold = min_count, smoothing = False, lex_only = True)
        
        if save_phraser == True:
            if language == 'en':
                common_terms = self.function_words_single
            else:
                common_terms = safe_get_stop_words(language)

            phrasegrams = {}
            for row in association_df.itertuples():
                word = row[1] + "_" + row[2]
                if row[3] > threshold:
                    phrasegrams[word] = row[3]
        
            phrases = Phrases(delimiter="_", connector_words=common_terms, min_count=min_count, threshold=threshold)
            phrases.phrasegrams = phrasegrams
            self.phrases = phrases
            
        return association_df
        
    def get_corpus_similarity(self, df1, df2, language='en'):
    
        cs = Similarity(language = self.settings.MAP_THREE[language])
        result = cs.calculate(self.read(df1), self.read(df2))
        
        return result

    def get_features(self, df, features="style"):
        """
        Extract feature vectors (x)

        :param df:
        :param features:
        :return:
        """
        # Function word ngrams
        if features == "style":
            x = self.style_vectorizer.transform(self.read(df))
            vocab_size = len(self.style_vectorizer.vocabulary_.keys())

        # Positive and negative words
        elif features == "sentiment":
            x = self.sentiment_vectorizer.transform(self.read(df))
            vocab_size = len(self.sentiment_vectorizer.vocabulary_.keys())
            
        # Constructions from the external C2xG package
        elif features == "constructions":
            cxg = C2xG(language = "eng")
            x = cxg.parse_return(input = self.read(df), mode = "lines", workers = 1)
            x = coo_matrix(x)
            vocab_size = len(cxg.model)
        
        # TF-IDF weighted content words, with PMI for phrases
        else:  # features == "content":
            x = self.tfidf_vectorizer.transform(self.read(df))
            vocab_size = len(self.tfidf_vectorizer.vocabulary_.keys())

        return x, vocab_size

    @staticmethod
    def split_data(df, test_size=0.10, n=2):
        """
        Train and test a Linear SVM classifier

        :param df:
        :param test_size:
        :param n:
        :return:
        """

        # In most cases, we just want training/testing data
        if n == 2:
            train_df, test_df = train_test_split(df, test_size=test_size)
            return train_df, test_df

        # If we're using an MLP, we might want training/testing/validation data
        elif n == 3:
            train_df, test_df = train_test_split(df, test_size=test_size + test_size)
            test_df, val_df = train_test_split(test_df, test_size=0.50)
            return train_df, test_df, val_df

    def fit_tfidf(self, df, n_features = 5000, min_count=1, language='en', force_phrases=False, vocab=None, workers=1):
        """
        Go through a dataset to build a content word vocabulary, with TF-IDF weighting
        :param df:
        :param min_count:
        :param language:
        :param vocab:
        :return:
        """

        # Get multi-word expressions using PMI with gensim
        self.fit_phrases(df, min_count=min_count, language=language, force=force_phrases)
        ai_logger.debug("Finished finding phrases.")

        vocab_list = self._get_vocab_list(df, n_features, min_count, language, vocab=vocab, workers=workers)
 
        # Initialize TF-IDF Vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            input="content",
            encoding="utf-8",
            decode_error="replace",
            ngram_range=(1, 1),
            analyzer=partial(clean, phraser=self.phrases),
            vocabulary=vocab_list,
            norm="l2",
            use_idf=True,
            smooth_idf=True,
            tokenizer=None,
        )

        # Fit on the dataset
        ai_logger.debug("Fitting TF-IDF")
        self.tfidf_vectorizer.fit(raw_documents=self.read(df))

    def _build_phrases(self, df, min_count = 1, language='en'):

        if language == 'en':
            common_terms = self.function_words_single
        else:
            common_terms = safe_get_stop_words(language)

        phrases = Phrases(
            sentences=stream_clean(df),
            min_count=min_count,
            threshold=0.70,
            scoring="npmi",
            max_vocab_size=20000000,
            delimiter="_",
            connector_words=common_terms
        )
        
        self.phrases = phrases

    def fit_phrases(self, df, min_count=1, language='en', force=False):
        """
        Use PMI to learn what phrases and collocations should be treated as one word
        :param df:
        :param min_count:
        :param language:
        :param save:
        :param filename:
        :return:
        """
        if not self.phrases or force:
            self._build_phrases(df, min_count, language)

    @staticmethod
    def _get_classifier(classifier):
       
        if classifier == 'svm':
            obj_class = LinearSVC(
                            penalty="l2",
                            loss="squared_hinge",
                            tol=0.0001,
                            C=1.0,
                            multi_class="ovr",
                            fit_intercept=True,
                            intercept_scaling=1,
                            max_iter=200000
                            )
        else:
            obj_class = LogisticRegression(penalty='l2',
                            tol=0.0001, 
                            C=1.0, 
                            fit_intercept=True, 
                            intercept_scaling=1, 
                            solver="lbfgs", 
                            max_iter=200000, 
                            multi_class="ovr", 
                            n_jobs=1, 
                            )
        
        return obj_class

    def shallow_classification(self, df, labels, features="style", cv=False, classifier='svm', x=None):
        """
        Train and test a Linear SVM classifier
        :param df:
        :param labels:
        :param features:
        :param cv:
        :param classifier:
        :return:
        """
        # TODO: saving model in self.classifier for a more pythonistic approach

        # Initialize the classifier
        self.classifier = self._get_classifier(classifier)

        # Use training/testing evaluation method
        if not cv:

            # Split into train/test
            train_df, test_df = self.split_data(df, test_size=0.10)

            # Get features
            train_x, vocab_size = self.get_features(train_df, features)
            test_x, vocab_size = self.get_features(test_df, features)

            # Train and save classifier
            self.classifier.fit(X=train_x, y=train_df.loc[:, labels].values)
            # self.cls = cls

            # Evaluate on test data
            predictions = self.classifier.predict(test_x)
            report = classification_report(y_true=test_df.loc[:, labels].values, y_pred=predictions)
            ai_logger.debug(report)
            result = report

        # Use 10-fold cross-validation for evaluation method
        else:
            # Get features
            x, vocab_size = self.get_features(df, features)

            # Run cross-validator
            scores = cross_validate(
                estimator=self.classifier,
                X=x,
                y=df.loc[:, labels].values,
                scoring=["precision_weighted", "recall_weighted", "f1_weighted"],
                cv=10,
                return_estimator=False,
            )

            # Show results; we can't save the classifier because we trained 10 different times
            ai_logger.debug(scores)
            result = scores

        return result

    def mlp(self, df, labels, features="style", validation_set=False, test_size=0.10, x=None):
        """
        Train and test a Multi-Layer Perceptron classifier (only works for non-binary classes)

        :param df:
        :param labels:
        :param features:
        :param x: For testing purposes.
        :param validation_set:
        :param test_size:
        :return:
        """
        # Make train/test split

        val_df = None
        val_x = None

        if not validation_set:
            train_df, test_df = self.split_data(df, test_size=test_size)

        # Make train / test / validation split
        else:
            train_df, test_df, val_df = self.split_data(df, test_size=test_size, n=3)

        # Find the number of classes
        n_labels = len(list(set(train_df.loc[:, labels].values)))

        # TensorFlow requires one-hot encoded labels (not strings)
        labeler = LabelEncoder()
        y_train = labeler.fit_transform(train_df.loc[:, labels].values)
        y_test = labeler.transform(test_df.loc[:, labels].values)

        if validation_set:
            y_val = labeler.transform(val_df.loc[:, labels].values)
        else:
            # Get y_val if necessary
            y_val = y_test

        # Feature extraction
        train_x, vocab_size = self.get_features(train_df, features)
        test_x, vocab_size = self.get_features(test_df, features)

        if validation_set:
            val_x, vocab_size = self.get_features(val_df, features)

        # Initializing the model
        model = tf.keras.Sequential()
        model.add(tf.keras.Input(shape=(vocab_size,)))

        # One or more dense layers.
        for units in [50, 50, 50]:
            model.add(tf.keras.layers.Dense(units, activation="relu"))

        # Output layer. The first argument is the number of labels.
        # Its sigmoid when binary. if sigmoid n_labels = 1
        # Its softmax when multi class, leave n_labels alone.
        if n_labels == 1:
            model.add(tf.keras.layers.Dense(n_labels, activation="sigmoid"))
            loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        else:
            model.add(tf.keras.layers.Dense(n_labels, activation="softmax"))
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        # Compile model
        model.compile(optimizer="adam",
                      loss=loss,
                      metrics=["accuracy"]
                      )

        # Now, begin or resume training
        model.fit(x=train_x.todense(),
                  y=y_train,
                  validation_data=(test_x.todense(), y_test),
                  epochs=10,
                  use_multiprocessing=True,
                  workers=5,
                  )

        # Reuse testing data if no validation set
        if not validation_set:
            val_x = test_x
            val_y = y_test
            val_y = labeler.inverse_transform(val_y)
        else:
            val_y = labeler.inverse_transform(y_val)

        # Evaluate on held-out data; TensorFlow returns probabilities, not classes
        # if binary don't do argmax, just return the prediction
        if n_labels == 1:
            y_predict = model.predict(val_x.todense())
        else:
            y_predict = np.argmax(model.predict(val_x.todense()), axis=-1)

        # Turn classes into string labels
        y_predict = labeler.inverse_transform(y_predict)

        # Get evaluation report
        report = classification_report(y_true=val_y, y_pred=y_predict)
        ai_logger.debug(report)

        # Save to class object
        self.model = model

    def df_to_index(self, df):
        """
        Get embedding indexes for whole dataframe
        :param df:
        :return:
        """
        x = np.array([line_to_index(line, self.max_size, self.word_vectors_vocab, self._nlp)
                      for line in df.loc[:, "Text"].values])
        return x

    def _make_sequential_model(self):
        model = tf.keras.Sequential()

        # Create a tensorflow layer from our word2vec embeddings
        embedding_layer = tf.keras.layers.Embedding(input_dim=self.word_vectors.shape[0],
                                                    output_dim=self.word_vectors.shape[1],
                                                    weights=[self.word_vectors],
                                                    input_length=self.max_size,
                                                    )

        # Add embedding layer as first layer in model
        model.add(embedding_layer)

        # The embedding layer needs to be flattened to fit into an MLP
        model.add(tf.keras.layers.Flatten())

        # One or more dense layers
        for units in [500, 500, 500]:
            add_layer = tf.keras.layers.Dense(units, activation="relu")
            model.add(add_layer)

        # Drop out layer, avoid over-fitting
        dropout_layer = tf.keras.layers.Dropout(0.2)
        model.add(dropout_layer)

        # Output layer. The first argument is the number of labels
        output_layer = tf.keras.layers.Dense(1, activation="sigmoid")
        model.add(output_layer)

        # Compile model
        model.compile(optimizer="adam",
                      loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=["accuracy"]
                      )
        return model

    def mlp_embeddings(self, df, labels, x, model=None):
        """
        A Multi-layer perceptron with embeddings as input (only works for binary classes)

        :param df:
        :param labels:
        :param x:
        :param model:
        :return:
        """
        # TensorFlow requires encoded labels (not strings)
        labeler = LabelEncoder()
        y = df.loc[:, labels].values.reshape(-1, 1)
        y = labeler.fit_transform(y)

        # Get train/test split
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10)

        # Find the number of classes
        # TODO: This variable is never used, what do we do?
        n_labels = len(list(set(df.loc[:, labels].values)))

        # If there's no model already, make one
        if not model:
            model = self._make_sequential_model()

        # Now, begin or resume training
        model.fit(x=x_train,
                  y=y_train,
                  validation_data=(x_test, y_test),
                  epochs=25,
                  use_multiprocessing=True,
                  )

        # Evaluate on held-out data; TensorFlow returns a sigmoid function, not classes
        y_predict = model.predict(x_test)
        y_predict = (y_predict[:, 0] > 0.5).astype(np.int32)

        # Turn classes into string labels
        y_predict = labeler.inverse_transform(y_predict)
        y_test = labeler.inverse_transform(y_test)

        # Get evaluation report
        report = classification_report(y_true=y_test, y_pred=y_predict)
        ai_logger.debug(report)

        return model

    def wordclouds(self, df, stage=0, features="frequency", name=None, stopwords=None):
        """
        Build wordclouds and choose what features to use
        :param df:
        :param stage:
        :param features:
        :param name:
        :param stopwords:
        :return:
        """
        # If only using frequency, use a pure Python method
        if features == "frequency":
            vocab = self._get_wordcloud_frequency_vocab(df, stage)
        # If using TF-IDF, use pre-fit vectorizer
        else:
            # features == "tfidf":
            vocab = self._get_wordcloud_tfidf_vocab(df)

        # Remove defined stopwords
        if stopwords is not None:
            vocab = ct.keyfilter(lambda x: x not in stopwords, vocab)

        # Pass pre-made frequencies to wordcloud, allowing for TF-IDF
        self.wordcloud.generate_from_frequencies(frequencies=vocab)

        # Prepare plot with title, etc
        self._plot_wordcloud(stage, name)

    @staticmethod
    def cluster(x, y="Missing", k=None, ari=True):
        """
        Use K-Means clustering
        :param x:
        :param y:
        :param k:
        :param ari:
        :return:
        """
        # If necessary, set k to the number of unique labels
        if not k:
            try:
                k = len(list(set(y)))
                # TODO: What exception can arise here?
            except Exception:
                k = 10
            ai_logger.debug("Using k=" + str(k))

        # Initiate KMeans clustering
        cluster = KMeans(n_clusters=k,
                         init="k-means++",
                         n_init=10,
                         max_iter=10000,
                         tol=0.0001,
                         copy_x=False,
                         algorithm="full")


        # Get cluster assignments
        clustering = cluster.fit_predict(X=x)

        # Set a null y if necessary; this allows us to cluster without known class labels
        try:
            test = y.shape
        except AttributeError:
            if y == "Missing":
                y = [0 for _ in range(len(x))]

        # Make a DataFrame showing the label and the cluster assignment
        cluster_df = pd.DataFrame([y, clustering]).T
        cluster_df.columns = ["Label", "Cluster"]

        if ari:
            ari = adjusted_rand_score(cluster_df.loc[:, "Label"].values, cluster_df.loc[:, "Cluster"].values)
            return ari, cluster_df
        else:
            return cluster_df

    @staticmethod
    def linguistic_distance(x, y, sample=1, n=1):
        """
        Manipulate linguistic distance to find the nearest examples
        :param x:
        :param y:
        :param sample:
        :param n:
        :return:
        """

        # Get the vector that represents our sample
        x_sample = x[sample]

        # Make sure we're using a dense matrix
        if isspmatrix(x_sample):
            x_sample = x_sample.todense()

        # We get each distance as we go
        holder = []

        # Compare each vector with the sample
        for i in range(x.shape[0]):
            if i != sample:
                x_test = x[i]

                # Make sure we're using a dense matrix
                if isspmatrix(x_test):
                    x_test = x_test.todense()

                # Calculate distance
                distance = euclidean(x_sample, x_test)

                # Add index and distance
                holder.append([i, distance])

        # Make a dataframe with all distances and sort, smallest to largest
        distance_df = pd.DataFrame(holder, columns=["Index", "Distance"])
        distance_df.sort_values(by="Distance", axis=0, ascending=True, inplace=True)

        # Reduce to desired number of comparisons
        distance_df = distance_df.head(n)

        # Get the labels for the sample and the closest document
        y_sample = y[sample]
        y_closest = [y[x] for x in distance_df.loc[:, "Index"].values]

        return y_sample, y_closest

    def train_word2vec(self, df, min_count=None, workers=1, language='en'):
        """
        Learn a word2vec embeddings from input data using gensim

        :param df:
        :param min_count:
        :return:
        """

        # If no min_count, find one
        if min_count is None:
            min_count = self.get_min_count(df)

        # If we haven' t learned phrases yet, do that now
        self.fit_phrases(df=df, min_count=min_count, language=language)
        
        data = [x for x in stream_clean(df, phraser=self.phrases, nlp=True, workers = workers)]

        # Learn the word embeddings
        embeddings = Word2Vec(
            sentences=data,
            vector_size=100,
            sg=1,
            window=4,
            hs=0,
            negative=20,
            min_count=min_count,
            epochs=10,
            workers=workers,
            max_vocab_size=20000000,
        )

        # Keep just the keyed vectors
        ai_logger.debug("Finished training")
        word_vectors = embeddings.wv.get_normed_vectors()
        print(word_vectors)

        vocab = embeddings.wv.key_to_index
        print(vocab)
        ai_logger.debug(word_vectors.shape)

        # Save to class
        self.word_vectors = word_vectors
        self.word_vectors_vocab = vocab

    def train_lda(self, df, n_topics, min_count=2):
        """
        Learn an LDA topic model from input data using gensim
        :param df:
        :param n_topics:
        :param min_count:
        :return:
        """
        

        cleaned_df = read_clean(df, phraser=self.phrases)
        # Get gensim dictionary, remove function words and infrequent words
        common_dictionary = Dictionary(cleaned_df)
        common_dictionary.filter_extremes(no_below=min_count)
        remove_ids = [common_dictionary.token2id[x] for x in self.function_words_single if
                      x in common_dictionary.token2id]

        # Filter out words we don't want
        common_dictionary.filter_tokens(bad_ids=remove_ids)
        common_corpus = [common_dictionary.doc2bow(text) for text in cleaned_df]

        # Train LDA
        lda = LdaModel(common_corpus, num_topics=n_topics)

        # Save to class
        self.lda = lda
        self.lda_dictionary = common_dictionary
        ai_logger.debug("Done learning LDA model")

    def use_lda(self, df, labels):
        """
        Get a fixed minimum frequency threshold based on the size of the current data set

        :param df:
        :param labels:
        :return:
        """
        # Get the gensim representation
        cleaned_df = read_clean(df, phraser=self.phrases)
        corpus = [self.lda_dictionary.doc2bow(text) for text in cleaned_df]

        # Get labels
        y = df.loc[:, labels].values

        # For storing topic results
        holder = []

        # Process each sample
        for i in range(len(corpus)):
            vector = self.lda[corpus[i]]
            label = y[i]
            main = 0.0
            main_index = 0

            # Find most relevant topic
            for cluster, val in vector:
                if val > main:
                    main_index = cluster
                    main = val

            holder.append([i, label, main_index])

        topic_df = pd.DataFrame(holder, columns=["Index", "Class", "Topic"])
        return topic_df

    @staticmethod
    def get_min_count(df):
        """
        Get a fixed minimum frequency threshold based on the size of the current data set
        :param df:
        :return:
        """
        if len(df) < 10000:
            min_count = 5
        else:
            min_count = int(len(df) / 10000)
        return max(min_count, 5)

    @staticmethod
    def print_sample(df):
        """
        Print just one text from a data set
        :param df:
        :return:
        """
        line = df.sample().loc[:, "Text"].values
        ai_logger.debug(line)
        return line[0]

    @staticmethod
    def print_labels(df, labels):
        """
        Print an inventory of labels counts, and return it as a dictionary

        :param df:
        :param labels:
        :return:
        """
        return ct.frequencies(df.loc[:, labels])

    @staticmethod
    def print_vector(vector, vectorizer):
        """
        Transform a sparse vector into a series with word labels

        :param vector:
        :param vectorizer:
        :return:
        """
        columns = [k for k, v in sorted(vectorizer.vocabulary_.items(), key=lambda item: item[1])]
        vector = pd.DataFrame(vector.todense()[0], columns=columns).T
        return vector

    def unmasking(self, df, labels, features):
        """
        Transform a sparse vector into a series with word labels
        :param df:
        :param labels:
        :param features:
        :return:
        """
        # Split into train/test
        train_df, test_df = train_test_split(df, test_size=0.10)

        # Get features
        train_x, vocab_size = self.get_features(train_df, features=features)
        test_x, vocab_size = self.get_features(test_df, features=features)

        # Make dense feature vectors
        train_x = pd.DataFrame(train_x.todense())
        test_x = pd.DataFrame(test_x.todense())
        ai_logger.debug(len(train_x.columns))

        # Iterate over 100 rounds of feature pruning
        for i in range(0, 100):

            # Initialize the classifier
            cls = LinearSVC(
                penalty="l2",
                loss="squared_hinge",
                dual=True,
                tol=0.0001,
                C=1.0,
                multi_class="ovr",
                fit_intercept=True,
                intercept_scaling=1,
                max_iter=200000
            )

            # Train and save classifier
            cls.fit(X=train_x, y=train_df.loc[:, labels].values)

            # Evaluate on test data
            predictions = cls.predict(test_x)
            report = classification_report(y_true=test_df.loc[:, labels].values, y_pred=predictions)
            ai_logger.debug(report)

            # The features to drop
            to_drop = []

            # Get most predictive features
            weights = pd.DataFrame(cls.coef_)

            # Look for highest feature for each class
            for index, row in weights.iterrows():
                max_index = row.idxmax()
                max_index = train_x.columns[max_index]
                to_drop.append(max_index)

            # Now remove features and start again
            train_x.drop(columns=to_drop, inplace=True)
            test_x.drop(columns=to_drop, inplace=True)
            ai_logger.debug(len(train_x.columns))
