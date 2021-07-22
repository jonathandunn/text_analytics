from text_analytics import TextAnalytics
import pandas as pd

files = ["test_data.gz"]

for file in files:

    ai = TextAnalytics()
    df = pd.read_csv(file)
    df = df.head(100)
    print(df)

    vocab = ai._get_vocab_list(df, min_count = 1, language = "non", return_freq = True)
    
    print(vocab)
    
    df = pd.DataFrame.from_dict(vocab, orient = "index", columns = ["Freq"])
    print(df)
    df = df.sort_values(by = "Freq", ascending = False)
    print(df)