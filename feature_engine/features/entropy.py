#Get priors using the previous conversations. and try to check how 

import pandas as pd
import scipy as sp
from scipy.stats import entropy
import nltk
import numpy as np
from nltk import ngrams
from collections import Counter
import math
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from nltk.tokenize import word_tokenize
import itertools

def ngram_dialog_act_entropy(df,on_column,n,set1,set2,set1_label,set2_label):

    # Initialize count vectorizer - this will extract ngrams from 1 to n (this covers maximum words/combinations possible)
    vectorizer = CountVectorizer(vocabulary=set1+set2)
    print(vectorizer)

    # Count occurrences of words in each bag. 
    #Each row of the matrix corresponds to a chat
    # and each column corresponds to a word or n-gram from the vocabulary.
    X = vectorizer.fit_transform(df[on_column])
    
    #convert sparse matrix to dense matrix
    X_dense = X.toarray() 
    
    #get the counts of the words in the matrix
    set1_counts = [row[:len(set1)] for row in X_dense]
    set2_counts = [row[-len(set2):] for row in X_dense]

    #Structure Set 1 - 0 3 1 - means  sentence 1 has 0 words from set 1, sentence 2 has 3 words, sentence 3 has 1 word 
    set1_totals = np.sum(set1_counts,axis = 1)
    set2_totals = np.sum(set2_counts,axis = 1)
    
    # Normalize counts to obtain probabilities
    set1_probs = set1_totals/np.sum(set1_totals)
    set2_probs = set2_totals/np.sum(set2_totals)

    #create an input matrix for the scikit entropy function
    entropy_matrix = []

    for i in range(len(set1_probs)):
        #create the array
        sentence_array = np.array([set1_probs[i],set2_probs[i]])

        probability = np.array(sentence_array)  
        entropy_val = entropy(probability, base=2)
        entropy_matrix.append(entropy_val)

        df.at[i,"entropy"] = entropy_val
