from gensim.models import word2vec
import pickle
import sys
from config import TRAIN_DATA_PATH, WORDS_LIST

def training_data(train_data):
    model = word2vec.Word2Vec(train_data,
                        size=100,
                        min_count=5,
                        window=5,
                        iter=20,
                        sg=0)
    return model

if __name__ == "__main__":
    with open(TRAIN_DATA_PATH, mode='rb') as f:
        train_data = pickle.load(f)

    model = training_data(train_data)
    args = sys.argv
    with open(WORDS_LIST, mode='rb') as f:
        words_list = pickle.load(f)


    search_words = []
    for search_word in args[1:]:
        if search_word in words_list:
            search_words.append(search_word)
    a = model.most_similar(positive=search_words)
    print(a)
    """
    a = model.most_similar(args[1])
    print(a)"""
