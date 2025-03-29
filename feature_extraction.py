from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import gensim
import numpy as np


def get_sentence_vector(sentence, model):
    vectors = [model.wv[word] for word in sentence if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

def feature_extraction_fun(processed_data):
    print("Feature Extraction Function")

    model= gensim.models.Word2Vec(processed_data, min_count=1,vector_size=100, window=5)
    
    sentence_vectors = np.array([get_sentence_vector(sentence, model) for sentence in processed_data])
    
    return sentence_vectors





