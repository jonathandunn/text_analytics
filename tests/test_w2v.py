from text_analytics import TextAnalytics

files = ["test_data.gz"]

for file in files:

    print(file)
    ai = TextAnalytics()
    
    phrases_file = file + ".phrases"
    phrases_file = "stylistics.gutenberg_all.gz.phrases"
    
    min_count = 250
    print(min_count)
    
    ai.train_word2vec(file, min_count, workers=30)
    
    word_vectors = ai.word_vectors
    word_vectors_vocab = ai.word_vectors_vocab
    
    print(word_vectors)
    print(word_vectors_vocab)
    
    ai.serialize(word_vectors, "w2v_embedding", file + ".w2v_embedding.json")
    ai.serialize(word_vectors_vocab, "w2v_vocab", file + ".w2v_vocab.json")
    
    word_vectors = ai.deserialize("w2v_embedding", file + ".w2v_embedding.json")
    word_vectors_vocab = ai.deserialize("w2v_vocab", file + ".w2v_vocab.json")
    
    print(word_vectors)
    print(word_vectors_vocab)