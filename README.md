# text_analytics

Basic computational linguistics and natural language processing in Python.

    pip install textanalytics
    
    pip install git+https://github.com/jonathandunn/text_analytics.git

This package provides code to support introductory courses in computational linguistics or natural language processing. These courses are available free on edX:

[**Introduction to Text Analytics and Natural Language Processing with Python**](https://www.edx.org/course/introduction-to-text-analytics-with-python)

[**Visualizing Text Analytics and Natural Language Processing with Python**](https://www.edx.org/course/visualizing-text-analytics-with-python)

## Usage

	from text_analytics import TextAnalytics
	
	ai = TextAnalytics()
	
## Getting features

	style, vocab_size = ai.get_features(df, features="style")
	
	*style* = Function word n-grams
	
	*sentiment* = Positive and negative words
	
	*content* = Top content words with TD-IDF weighting, PMI for finding phrases, no stop words
	
	*constructions* = A bag-of-constructions syntactic representation
	
## Using a classifier

	ai.shallow_classification(df, label, features="style", cv=False, classifier='svm')
	
	ai.mlp(df, label, features="style", validation_set=False, test_size=0.10)
	
#Unsupervised methods

	*Topic Models*

	ai.train_lda(df, n_topics, min_count)
        
    topic_df = ai.use_lda(df, labels="Author")
	
	*Vector Semantics*
	
	ai.train_word2vec(file, min_count, workers)
	
	*Document and Word Clusters*
	
	cluster_df = ai.cluster(x, y=None, k)
	
	*Nearest document searches
	
	 y_sample, y_closest = ai.linguistic_distance(x, y, sample=1, n=3)
	 
#Corpus Descriptions

	*PMI-based Phrases*
	
	ai.fit_phrases(df)
	 
	*Delta P-based Phrases*
	 
	association_df = ai.get_association(df, min_count = 1, save_phraser = True)
	 
	*Basic word frequencies*
	 
	vocab = ai._get_vocab_list(df, min_count, return_freq = True)
	 
	*Corpus Comparisons*
	
	similarity = ai.get_corpus_similarity(df1, df2)
	 
	 
