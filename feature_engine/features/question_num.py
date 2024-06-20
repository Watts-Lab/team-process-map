import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import re


def calculate_num_question_naive(text, question_words):
    """
    Get the number of sentences that either end with a question mark or start with a 
    question word (e.g., "who," "what," "when," "where," "why").
    This is a naive measure of the number of questions asked. 

    Note that this function requires the data to *retain* punctuation as input.

    Args:
        text (str): The message (utterance) being analyzed.
        question_words (list): The list of question words.

    Returns: 
        int: The number of questions (Sentences ending with question marks or starting with question words) in the text.
    """
    # step 1: tokenize sentence
    sentences = sent_tokenize(text)
    num_q = 0
    for sentence in sentences:
        # Only proceed if the sentence contains letters or numbers
        if re.match("^[a-zA-Z0-9 ]+", sentence):
            # Is a question if the sentence ends with "?" or starts with a word that is in the question_words list
            if sentence.endswith("?") or word_tokenize(sentence)[0] in question_words:
                num_q += 1
    return num_q