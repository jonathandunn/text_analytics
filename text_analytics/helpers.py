from text_analytics.settings import *
from gensim.parsing import preprocessing
from collections import defaultdict
import numpy as np
import cytoolz as ct
import re
import json
import spacy


def clean_web(line):
    """
    Cleans web info from corpus
    :param line:
    :return:
    """
    line = re.sub(r"http\S+", "", line)
    line = re.sub(r"@\S+", "", line)
    line = re.sub(r"#\S+", "", line)
    line = re.sub("<[^>]*>", "", line)
    line = line.replace(" RT", "").replace("RT ", "")
    return line


def clean(line, phraser=None, nlp=None):
    """
    Pre-processing function that splits words, gets phrases, removes stopwords
    :param line:
    :param phraser:
    :param nlp:
    :return:
    """
    line = clean_web(line)

    line = remove_punctuation(line)

    # Strip and lowercase
    line = line.lower().strip().lstrip().split()

    # If we've used PMI to find phrases, get those phrases now
    if phraser is not None:
        line = list(phraser[line])

    # If we want Part-of-Speech tagging, do that now
    if nlp:
        line = nlp(" ".join(line))
        line = [w.text + "_" + w.pos_ for w in line]

    return line


def read_clean(df, nlp=None, column="Text"):
    """
    Returns a list of cleaned strings from the dataframe

    :param df:
    :param nlp:
    :param column:
    :return:
    """
    return [clean(str(x), nlp=nlp) for x in df.loc[:, column].values]


def clean_pre(line):
    """

    Pre-processing function that doesn't strip words (for style and sentiment)
    :param line:
    :return:
    """
    # Remove links, hashtags, at-mentions, mark-up, and "RT"
    line = clean_web(line)

    # Remove punctuation and extra spaces
    line = remove_punctuation(line)

    # Strip and lowercase
    line = line.lower().strip().lstrip()

    return line


def remove_punctuation(line):
    """
    Removes punctuation from corpus
    :param line:
    :return:
    """
    return ct.pipe(line,
                   preprocessing.strip_tags,
                   preprocessing.strip_punctuation,
                   preprocessing.strip_numeric,
                   preprocessing.strip_non_alphanum,
                   preprocessing.strip_multiple_whitespaces
                   )


def nlp_tag(text, nlp=None):
    """
    Loads nlp tag from spacy library.
    :param text:
    :param nlp:
    :return:
    """
    if not nlp:
        nlp = spacy.load("en_core_web_sm")
    nlp.max_length = SPACY_MAX_LENGTH
    return nlp(text)


def clean_wordclouds(line, function_words_single, stage=1, phrases=None):
    """
        #Pre-processing function that splits words, gets phrases, removes stopwords
        ## 0=just split into words
        ## 1=remove stop words
        ## 2=lowercase
        ## 3=remove punctuation
        ## 4=remove non-linguistic material
        ## 5=join phrases
    :param line:
    :param function_words_single:
    :param stage:
    :param phrases:
    :return:
    """

    if stage > 3:
        # Remove links, hashtags, at-mentions, mark-up, and "RT"
        line = clean_web(line)

    if stage > 2:
        # Remove punctuation and extra spaces
        line = remove_punctuation(line)

    if stage > 1:
        # Strip and lowercase
        line = line.lower().strip().lstrip().split()
    else:
        line = line.split()

    if stage > 4:
        if phrases is not None:
            line = list(phrases[line])

    if stage > 0:
        line = [x for x in line if x not in function_words_single]

    return line


def line_to_index(line, max_size, word_vectors_vocab, nlp=None):
    """
    Go from the input line to a list of word2vec indexes
    :param line:
    :param max_size:
    :param word_vectors_vocab:
    :return:
    """
    # Get an empty list for indexes, clean the line, and prune to max size
    line_index = []
    line = clean(line, nlp=nlp)
    line = line[:max_size]

    # Get the embedding index for each word
    for word in line:
        try:
            line_index.append(word_vectors_vocab[word].index)
            # TODO: What exception can arise here?
        except Exception:
            pass

    # We need each speech to have the same dimensions, so we might need to add padding
    while len(line_index) < max_size:
        line_index.append(0)

    line_index = np.array(line_index)

    return line_index


def get_vocab(df):
    """
    Gets vocab
    :param df:
    :return:
    """
    vocab = defaultdict(int)
    cleaned_df = read_clean(df)
    for line in cleaned_df:
        for word in line:
            vocab[word] += 1
    return vocab


class NumpyEncoder(json.JSONEncoder):
    """
    Json encoder for Numpy Lists.
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

