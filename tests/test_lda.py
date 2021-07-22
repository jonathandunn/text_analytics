from text_analytics import TextAnalytics
import pandas as pd

files = ["test_data.gz"]

for file in files:

    print(file)
    ai = TextAnalytics()

    df = pd.read_csv(file)
    df = df.head(200)
    print(df)
    
    ai.train_lda(df, n_topics=10, min_count=5)
    lda_model = ai.lda
    lda_dictionary = ai.lda_dictionary
    
    ai.serialize(lda_model, "lda_model", file + ".lda_model.json")
    ai.serialize(lda_dictionary, "lda_dictionary", file + ".lda_dictionary.json")
    
    ai.lda = ai.deserialize("lda_model", file + ".lda_model.json")
    ai.lda_dictionary = ai.deserialize("lda_dictionary", file + ".lda_dictionary.json")
    
    topic_df = ai.use_lda(df, labels="Author")
    print(topic_df)