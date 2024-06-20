import convokit
import spacy
import pandas as pd
import re

from convokit import Corpus, Speaker, Utterance
from convokit import download
from convokit import TextParser

from convokit import PolitenessStrategies

ps = PolitenessStrategies()

# Note: if you get an error in which `en_core_web_sm` is not found, do the following: python3 -m spacy download en_core_web_sm

spacy_nlp = spacy.load("en_core_web_sm", disable=["ner"])

def get_politeness_strategies(text):
    """
    Using the ConvoKit politeness package, obtains politeness annotations of each message, with some fields 
    including HASHEDGE, Factuality, Deference, Gratitude, Apologizing, etc.

    Source: https://convokit.cornell.edu/documentation/politenessStrategies.html

    Args:
       text(str): The text of the utterance to be analyzed.

    Returns:
        dict: A dictionary containing the politeness strategies extracted, in a format as follows.
        These names are then cleaned up downstream in the ChatLevelFeaturesCalculator before being appended
        to the output dataframe.
    """
    if pd.isnull(text):
        text = ""
    utt = ps.transform_utterance(
        text, spacy_nlp=spacy_nlp
    )
    return(utt.meta["politeness_strategies"])