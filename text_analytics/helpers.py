from gensim.parsing import preprocessing
from collections import defaultdict
import cleantext
import pandas as pd
import numpy as np
import cytoolz as ct
import re
import json
import spacy
import multiprocessing as mp
from functools import partial

try:
    from settings import Settings
except:
    from .settings import Settings
    
#Initialize the settings module
settings = Settings()

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

def clean(line, phraser=None, nlp=None, stop=None):
    """
    Pre-processing function that splits words, gets phrases, removes stopwords
    :param line:
    :param phraser:
    :param nlp:
    :return:
    """
   #Use clean-text 
    line = cleantext.clean(line,
                    fix_unicode = True,
                    to_ascii = False,
                    lower = True,
                    no_line_breaks = True,
                    no_urls = True,
                    no_emails = True,
                    no_phone_numbers = True,
                    no_numbers = True,
                    no_digits = True,
                    no_currency_symbols = True,
                    no_punct = True,
                    replace_with_punct = "",
                    replace_with_url = "URL",
                    replace_with_email = "EMAIL",
                    replace_with_phone_number = "PHONE",
                    replace_with_number = "NUMBER",
                    replace_with_digit = "0",
                    replace_with_currency_symbol = "CURRENCY"
                    ).split()
    
    # If we've used PMI to find phrases, get those phrases now
    if phraser is not None:
        line = list(phraser[line])

        
    # If we want Part-of-Speech tagging, do that now
    if nlp is not None:
        line = nlp(" ".join(line))
        line = [w.text + "_" + w.pos_ for w in line]
        
    #Remove words (checking for possible pos tag)
    if stop is not None:
        line = [word for word in line if word[:word.rfind("_")] not in stop]

    return line


def read_clean(df, phraser=None, stop=None, nlp=None, column="Text"):
    """
    Returns a list of cleaned strings from the dataframe

    :param df:
    :param phraser:
    :param stop:
    :param nlp:
    :param column:
    :return:
    """
    
    #In case we pass a filename instead of a loaded dataframe
    if isinstance(df, str):
        df = pd.read_csv(df)
        
    return [clean(line, phraser=phraser, stop=stop, nlp=nlp) for line in df.loc[:, column].values]
    
def process_stream(line, phraser=None, stop=None, nlp=None):
    """
    Multiprocess a list of lines if desired

    :param lines:
    :param phraser:
    :param nlp:
    :return:
    """
    if nlp is not None:
        nlp = spacy.load("en_core_web_sm")
        nlp.max_length = 99999999

    return clean(line, phraser=phraser, stop=stop, nlp=nlp)
    
def stream_clean(df, phraser=None, stop=None, nlp=None, column="Text"):
    """
    Yields a list of cleaned strings from the dataframe; avoids holding everything in memory

    :param df:
    :param nlp:
    :param column:
    :return:
    """
    if nlp is not None:
        nlp = spacy.load("en_core_web_sm")
        nlp.max_length = 99999999
    
    #In case we pass a filename instead of a loaded dataframe
    if isinstance(df, str):
        for temp_df in pd.read_csv(df, iterator=True, chunksize=1000):
            print(temp_df)
            for line in temp_df.loc[:,column].values:
                yield clean(line, phraser=phraser, stop=stop, nlp=nlp)
            
    #In case we pass a dataframe
    else:
        for line in df.loc[:, column].values:
            line = clean(str(line), phraser=phraser, stop=stop, nlp=nlp)
            yield line
        
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


def line_to_index(line, max_size, word_vectors_vocab, phrases=None, nlp=None):
    """
    Go from the input line to a list of word2vec indexes
    :param line:
    :param max_size:
    :param word_vectors_vocab:
    :return:
    """
    # Get an empty list for indexes, clean the line, and prune to max size
    line_index = []
    line = clean(line, phraser=phrases, nlp=nlp)
    line = line[:max_size]

    # Get the embedding index for each word
    for word in line:
        line_index.append(word_vectors_vocab.get(word, 0))

    # We need each speech to have the same dimensions, so we might need to add padding
    while len(line_index) < max_size:
        line_index.append(0)

    return np.array(line_index)
    
def process_vocab(lines, phraser=None, stop=None, nlp=None):

    vocab = defaultdict(int)
    for line in lines:
        for word in clean(line, phraser=phraser, stop=stop, nlp=nlp):
            vocab[word] += 1
            
    return vocab

def get_vocab(df, phraser=None, stop=None, nlp=None, column = "Text", workers = 1):
    """
    Gets vocab
    :param df:
    :return:
    """
    chunksize = int(len(df)/workers)
    
    pool_instance = mp.Pool(processes = workers, maxtasksperchild = 1)
    vocab = pool_instance.map(partial(process_vocab, phraser=phraser, stop=stop, nlp=nlp), ct.partition(chunksize, df.loc[:, column].values), chunksize = 1)
    pool_instance.close()
    pool_instance.join()
    
    vocab = ct.merge_with(sum, vocab)
              
    return vocab


class NumpyEncoder(json.JSONEncoder):
    """
    Json encoder for Numpy Lists.
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)