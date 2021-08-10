from text_analytics import TextAnalytics
import pandas as pd

files = ["test_data.gz"]

for file in files:

    ai = text_analytics.TextAnalytics()
    df = pd.read_csv(file)
    df_small = df.head(10)
    df1 = df.head(200)
    df2 = df.tail(200)
    print(df)

    x, vocab_size = ai.get_features(df_small, features = "constructions")
    print(x)
    print(vocab_size)
    
    association_df = ai.get_association(df, min_count = 1, save_phraser = True)
    print(association_df)
    print(ai.phrases)
    
    similarity = ai.get_corpus_similarity(df1, df2)
    print(similarity)
    