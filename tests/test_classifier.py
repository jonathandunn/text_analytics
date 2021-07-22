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

    ai.shallow_classification(df, labels = label, features="style", cv=False, classifier='svm')
    ai.shallow_classification(df, labels = label, features="style", cv=True, classifier='svm')
    ai.shallow_classification(df, labels = label, features="style", cv=False, classifier='lr')
    ai.shallow_classification(df, labels = label, features="style", cv=True, classifier='lr')
    
    ai.shallow_classification(df, labels = label, features="sentiment", cv=False, classifier='svm')
    ai.shallow_classification(df, labels = label, features="sentiment", cv=True, classifier='svm')
    ai.shallow_classification(df, labels = label, features="sentiment", cv=False, classifier='lr')
    ai.shallow_classification(df, labels = label, features="sentiment", cv=True, classifier='lr')
    
    #ai.shallow_classification(df, labels = label, features="content", cv=False, classifier='svm')
    #ai.shallow_classification(df, labels = label, features="content", cv=True, classifier='svm')
    #ai.shallow_classification(df, labels = label, features="content", cv=False, classifier='lr')
    #ai.shallow_classification(df, labels = label, features="content", cv=True, classifier='lr')
    
    ai.mlp(df, labels = label, features="style", validation_set=False, test_size=0.10)
    ai.mlp(df, labels = label, features="style", validation_set=True, test_size=0.10)
    
    ai.mlp(df, labels = label, features="sentiment", validation_set=False, test_size=0.10)
    ai.mlp(df, labels = label, features="sentiment", validation_set=True, test_size=0.10)
    
    #ai.mlp(df, labels = label, features="content", validation_set=False, test_size=0.10)
    #ai.mlp(df, labels = label, features="content", validation_set=True, test_size=0.10)
    
    #ai.shallow_classification(df, labels = label, features="constructions", cv=False, classifier='lr')
    #ai.mlp(df, labels = label, features="constructions", validation_set=False, test_size=0.10)