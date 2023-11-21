import pandas as pd
import numpy as np
import re
import os

from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer, util

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax


model_vect = SentenceTransformer('all-MiniLM-L6-v2')
MODEL  = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model_bert = AutoModelForSequenceClassification.from_pretrained(MODEL)

# Check if embeddings exist
def check_embeddings(chat_data, vect_path, bert_path):
    
    # vect_output_path = re.sub('./feature_engine', '', vect_path)
    # bert_output_path = re.sub('./feature_engine', '', bert_path)

    # ../feature_engine/embeddings --> ../embeddings
    if (not os.path.isfile(vect_path)):
        generate_vect(chat_data, vect_path)
    if (not os.path.isfile(bert_path)):
        generate_bert(chat_data, bert_path)

# Generate sentence vectors
def generate_vect(chat_data, output_path):

    print(f"Generating sentence vectors....")
    # print(f"This is the current filepath: {os. getcwd()}")
    # print(f"And we want to get to {output_path}")
    embedding_arr = [row.tolist() for row in model_vect.encode(chat_data.message)]
    embedding_df = pd.DataFrame({'message': chat_data.message, 'message_embedding': embedding_arr})


    # Create directories along the path if they don't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    embedding_df.to_csv(output_path)

# Generate BERT sentiments 
def generate_bert(chat_data, output_path):
    
    print(f"Generating BERT sentiments....")

    messages = chat_data['message']
    sentiments = messages.apply(get_sentiment)

    sent_arr = [list(dict.values()) for dict in sentiments]

    sent_df = pd.DataFrame(sent_arr, columns =['positive_bert', 'negative_bert', 'neutral_bert']) 
    
    # Create directories along the path if they don't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sent_df.to_csv(output_path)

def get_sentiment(text):

    if (pd.isnull(text)):
        return({'positive': np.nan, 'negative': np.nan, 'neutral': np.nan})
    
    encoded = tokenizer(text, return_tensors='pt')
    output = model_bert(**encoded)

    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    # sample output format
    return({'positive': scores[2], 'negative': scores[0], 'neutral': scores[1]})
