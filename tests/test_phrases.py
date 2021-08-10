from text_analytics import TextAnalytics
import pandas as pd

files = ["test_data.gz"]

for file in files:

    print(file)
    ai = TextAnalytics()

    df = pd.read_csv(file)
    df = df.head(100)
    print(df)
    
    style, vocab_size = ai.get_features(df, features="style")
    print(style)
    print(vocab_size)
    
    sentiment, vocab_size = ai.get_features(df, features="sentiment")
    print(sentiment)
    print(vocab_size)
    
    ai.fit_phrases(df)
    phrases = ai.phrases
    print(phrases)
    ai.serialize(phrases, "phrases", file + ".phrases.json")
    
    ai.fit_tfidf(df, n_features = 10000)
    x, vocab_size = ai.get_features(df, features = "content")
    print(x)
    print(vocab_size)
    ai.serialize(ai.tfidf_vectorizer, "tfidf_model", file + ".tfidf.json")
    
    ai.phrases = ai.deserialize("phrases", file + ".phrases.json")
    print(ai.phrases)
    
    ai.tfidf_vectorizer = ai.deserialize("tfidf_model", file + ".tfidf.json")
    print(ai.tfidf_vectorizer.vocabulary_)
    print(ai.tfidf_vectorizer)
    
    

    