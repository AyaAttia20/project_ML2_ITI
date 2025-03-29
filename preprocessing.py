import nltk
from nltk.tokenize import  word_tokenize
from IPython.display import display
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.corpus import wordnet
import re

#pos tags function
def get_wordnet_pos(tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith( 'N'):
         return wordnet.NOUN
        elif tag.startswith('R'):
         return wordnet.ADV
        else:
         return None
               
#preprocessing function
def preprocess(data):
    # print("preprocessing modeule for text data")
    # print("--------------------------------------------------------------------------------------")

    if isinstance(data, list):
        data = " ".join(data) 
    #tokenize the words
    words_tokens = word_tokenize(data)
    # display(f"Tokenized words: {words_tokens}")
    # print("--------------------------------------------------------------------------------------")


    #remove punctuations
    words_without_punct = [word.translate(str.maketrans('', '', string.punctuation)) for word in words_tokens]

    # display(f"words without punct: {words_without_punct}")
    # print("--------------------------------------------------------------------------------------")
    

    #remove stopwords and lowerning 
    filters = set(stopwords.words("english"))
    cleaned = [word.lower() for word in words_without_punct if word.lower() not in filters]
    # print("Words after removing stopwords and lowering : ", cleaned)
    # print("--------------------------------------------------------------------------------------")

    #lemmetization and pos tags
    postags=pos_tag(cleaned)
    # display(f"POS Tags: {postags}")
    # print("--------------------------------------------------------------------------------------")

    lemmatizer = WordNetLemmatizer()
    lemmatized_sentence = []  
    for word, tag in postags:
        wordnet_pos = get_wordnet_pos(tag) or wordnet.NOUN
        lemmatized_sentence. append(lemmatizer.lemmatize(word, pos=wordnet_pos))

    # display(f"Lemmatized words: {lemmatized_sentence}")    
    # print("--------------------------------------------------------------------------------------")


    #remove numbers
    removed_num= [re.sub(r'\d+', '', word) for word in lemmatized_sentence]
    # print("Text after removing numbers: ", removed_num)

    # print("--------------------------------------------------------------------------------------")


    #remove empty strings and extra spaces
    text = [word for word in removed_num if word.strip()] 

    # print("Final text after preprocessing: ", text)

    # print("--------------------------------------------------------------------------------------")
    removed_Special= [re.sub(r'[^\w\s]', '', word)  for word in text]
    return removed_Special
    # print("Text after removing numbers: ", removed_Special)

    
    






    

