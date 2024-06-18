import pandas as pd
import numpy as np
import re
import os
import pickle

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
    if (not os.path.isfile(Path(__file__).resolve().parent.parent/"features/lexicons/certainty.txt")):
        # unpickle certainty
        unpickle_certainty()


    ### TODO --- TEST THIS!
    current_script_directory = Path(__file__).resolve().parent
    LEXICON_PATH_STATIC = current_script_directory.parent/"features/lexicons_dict.pkl"
    if (not os.path.isfile(LEXICON_PATH_STATIC)):
        generate_lexicon_pkl()

# Read in the lexicons (helper function for generating the pickle file)
def read_in_lexicons(directory, lexicons_dict):
    for filename in os.listdir(directory):
        with open(directory/filename, encoding = "mac_roman") as lexicons:
            if filename.startswith("."):
                continue
            lines = []
            for lexicon in lexicons:
                # get rid of parentheses
                lexicon = lexicon.strip()
                lexicon = lexicon.replace('(', '')
                lexicon = lexicon.replace(')', '')
                if '*' not in lexicon:
                    lines.append(r"\b" + lexicon.replace("\n", "") + r"\b")
                else:
                    # get rid of any cases of multiple repeat -- e.g., '**'
                    lexicon = lexicon.replace('\**', '\*')

                    # build the final lexicon
                    lines.append(r"\b" + lexicon.replace("\n", "").replace("*", "") + r"\S*\b")
        clean_name = re.sub('.txt', '', filename)
        lexicons_dict[clean_name] = "|".join(lines)

# Generate the lexicon .pkl file
def generate_lexicon_pkl():
    print("Generating Lexicon pickle...")
    lexicons_dict = {}
    current_script_directory = Path(__file__).resolve().parent
    read_in_lexicons(current_script_directory.parent / "features/lexicons/liwc_lexicons/", lexicons_dict) # Reads in LIWC Lexicons
    read_in_lexicons(current_script_directory.parent / "features/lexicons/other_lexicons/", lexicons_dict) # Reads in Other Lexicons

    # Save as pickle
    with open(current_script_directory.parent/"features/lexicons_dict.pkl", "wb") as lexicons_pickle_file:
        pickle.dump(lexicons_dict, lexicons_pickle_file)

# Generate sentence vectors
def generate_vect(chat_data, output_path):

    print(f"Generating sentence vectors....")
    # print(f"This is the current filepath: {os. getcwd()}")
    # print(f"And we want to get to {output_path}")
    embedding_arr = [row.tolist() for row in model_vect.encode(chat_data.message)]
    embedding_df = pd.DataFrame({'message': chat_data.message, 'message_embedding': embedding_arr})


    # Create directories along the path if they don't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    embedding_df.to_csv(output_path, index=False)

# Generate BERT sentiments 
def generate_bert(chat_data, output_path):
    
    print(f"Generating BERT sentiments....")

    messages = chat_data['message']
    sentiments = messages.apply(get_sentiment)

    sent_arr = [list(dict.values()) for dict in sentiments]

    sent_df = pd.DataFrame(sent_arr, columns =['positive_bert', 'negative_bert', 'neutral_bert']) 
    
    # Create directories along the path if they don't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sent_df.to_csv(output_path, index=False)

def get_sentiment(text):

    if (pd.isnull(text)):
        return({'positive': np.nan, 'negative': np.nan, 'neutral': np.nan})
    
    text = ' '.join(text.split()[:512]) # handle cases when the text is too long: just take the first 512 chars (hacky, but BERT context window cannot be changed)
    encoded = tokenizer(text, return_tensors='pt')
    output = model_bert(**encoded)

    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    # sample output format
    return({'positive': scores[2], 'negative': scores[0], 'neutral': scores[1]})


def unpickle_certainty():
    current_script_directory = Path(__file__).resolve().parent

    with open(current_script_directory.parent/ "features/lexicons/certainty.pkl", "rb") as file:
        unpickled_content = pickle.load(file)

    with open(current_script_directory.parent/ "features/lexicons/certainty.txt", "w") as file:
        file.write(unpickled_content)
