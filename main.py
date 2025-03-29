import sys
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from IPython.display import display
from preprocessing import preprocess
import re
from sklearn.datasets import fetch_20newsgroups
import pandas as pd
import numpy as np
from feature_extraction import feature_extraction_fun
from clustering_module import pca_fun
from sklearn.decomposition import PCA
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


# try with simple data 
# processed_data=preprocess("This ?is a ! @test sentence?? running for testing $the preprocessing #module for 10 times,i went to    sleep at 10:00 PM and wek up at 5:00 AM,i working on module two now ")
# processed_text = ' '.join(processed_data)
# print(processed_text)
# print("--------------------------------------------------------------------------------------")
# feature_extraction_data=feature_extraction_fun([processed_text])

# print("--------------------------------------------------------------------------------------")
# print("Loading people_wiki dataset")

people_wiki_df = pd.read_csv("people_wiki.csv")

preprocessed=people_wiki_df['text'].apply(preprocess)


feature_extraction_data = feature_extraction_fun(preprocessed)

print("feature extraction successfully")

print("Shape of sentence_vectors:", feature_extraction_data.shape)

pca = PCA(n_components=2)
pca_features = pca.fit_transform(feature_extraction_data)
per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(per_var) + 1), per_var.cumsum(), marker="o", linestyle="--")
plt.grid()
plt.ylabel("Cumulative Percentage of Explained Variance")
plt.xlabel("Number of Components")
plt.title("Explained Variance by Component")
plt.show()







