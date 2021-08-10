from text_analytics import TextAnalytics
import pandas as pd

files = ["test_data.gz"]

for file in files:

    print(file)
    ai = TextAnalytics()

    df = pd.read_csv(file)
    df1 = df.tail(10)
    df2 = df.head(10)
    df = pd.concat([df1, df2], axis = 0)
    print(df)
    
    label = "Author"
    print(set(df.loc[:,label].values))

    style, vocab_size = ai.get_features(df, features="style")
    print(style)
    print(vocab_size)
    
    sentiment, vocab_size = ai.get_features(df, features="sentiment")
    print(sentiment)
    print(vocab_size)
    
    ari, cluster_df = ai.cluster(x=style, y=df.loc[:,"Author"].values, k=None, ari=True)
    print(cluster_df)
    print(ari)
    
    ari, cluster_df = ai.cluster(x=sentiment, y=df.loc[:,"Author"].values, k=None, ari=True)
    print(cluster_df)
    print(ari)
    
    y_sample, y_closest = ai.linguistic_distance(x=style, y=df.loc[:,"Author"].values, sample=1, n=3)
    print(y_sample)
    print(y_closest)
    
    y_sample, y_closest = ai.linguistic_distance(x=sentiment, y=df.loc[:,"Author"].values, sample=1, n=3)
    print(y_sample)
    print(y_closest)  