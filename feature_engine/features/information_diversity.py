import pandas as pd
import numpy as np
import math
import nltk
from nltk.corpus import stopwords
from nltk import tokenize
nltk.download('stopwords')
nltk.download('wordnet')
stopword = list(stopwords.words('english'))

from nltk.stem import WordNetLemmatizer  
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from pprint import pprint
from scipy.spatial.distance import cosine

import gensim.corpora as corpora
from gensim.models import CoherenceModel
from gensim.utils import simple_preprocess
from gensim.models.ldamodel import LdaModel

def get_info_diversity(df):
    """
    Computes information diversity (value between 0 and 1 inclusive) for all conversations.

    Args:
        df (pd.DataFrame): The utterance (chat)-level dataframe.
    
    Returns:
        pd.DataFrame: the grouped conversational dataframe, with a new column ("info_diversity") representing the conversation's information diversity score.
    """
    info_div_score = df.groupby("conversation_num").apply(lambda x : info_diversity(x)).reset_index().rename(columns={0: "info_diversity"})
    return info_div_score

def info_diversity(df):
    """
    Preprocess data and then create numeric mapping of words in dataset to pass into LDA model
    Uses square root of number of rows as number of topics

    Args:
        df (pd.DataFrame): The input dataframe, grouped by the conversation index, to which this function is being applied.
    
    Returns:
        float: The information diversity score, obtained from calling calculate_ID_score on the chat's topics; defaults to zero in case of empty data
    """
    num_rows = len(df)
    num_topics = int(math.sqrt(num_rows))
    processed_data = df['message'].apply(preprocessing).tolist()

    if not processed_data:
         return 0

    mapping = corpora.Dictionary(processed_data)
    full_corpus = [mapping.doc2bow(text) for text in processed_data]

    if (not full_corpus or not mapping):
         return 0
    else:
        lda = LdaModel(corpus=full_corpus, id2word=mapping, num_topics=num_topics)
        topics = [lda.get_document_topics(bow) for bow in full_corpus]
        ID = calculate_ID_score(topics, num_topics)
        return ID

def preprocessing(data):
        """
        Preprocesses the data by lowercasing, lemmatizing, and removing words of size less than 4

        Args:
            data (str): The utterance being analyzed (in this case, preprocessed for the LDA model.)

        Returns:
            list: A list of lemmatized text with stopwords and shorter words removed.
        """
        le=WordNetLemmatizer()
        word_tokens=word_tokenize(data.lower())
        tokens=[le.lemmatize(w) for w in word_tokens if w not in stopword and len(w) > 3]
        return tokens

def calculate_ID_score(doc_topics, num_topics):
        """
        Computes info diversity score as suggested in Reidl & Woolley (2017); determines a topic vector 
        for every message using an LDA Model, computes a mean topic vector across all messages, and measures the average 
        cosine similarity between the message vectors and the mean vector.

        Source: https://www.circlelytics.com/wp-content/uploads/2022/05/Riedl-Woolley-2017-Teams-vs-Crowds-A-field-test-of-the-realitive-contribution-of-incentives-member-abilities.pdf

        Args:
            doc_topics (list): the list of topic vectors from the team's chat messages that comes from the LDA model.
            num_topics (int): the number of topics; set to be the square root of the number of rows, rounded to the nearest integer (this is a design decision on our part to be robust to datasets of varying sizes).
        
        Returns:
            float: The information diversity score, given the list of topics vectors and the number of topics
            
        """
        topic_matrix = []
        for doc in doc_topics:
            topic_dist = np.zeros(num_topics)
            for topic, prob in doc:
                topic_dist[topic] = prob
            topic_matrix.append(topic_dist)
        topic_matrix = np.array(topic_matrix)

        mean_topic_vector = np.mean(topic_matrix, axis=0)
        squared_cosine_distances = [(1 - cosine(doc, mean_topic_vector))**2 for doc in topic_matrix]
        score = np.sum(squared_cosine_distances) / len(squared_cosine_distances)
        return score