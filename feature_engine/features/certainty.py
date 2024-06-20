import pandas as pd
import re
import os
import io
from pathlib import Path
import pickle

# Note: This feature requires the message WITH punctuation.

# parse certainty lexicon, compile into master regex, delimited by | 
# Construct the absolute path to certainty.txt using the current script directory
current_script_directory = Path(__file__).resolve().parent
certainty_file_pkl_path = current_script_directory / "lexicons/certainty.pkl"
with open(certainty_file_pkl_path, 'rb') as f:
    certainty_data = pickle.load(f)  # Load pickled data
    certainty = pd.read_csv(io.StringIO(certainty_data), sep = ",")
    certainty = certainty.sort_values(["NumWords", "NumCharacters"], ascending=False)
master_regex = certainty["Word"].str.cat(sep='\\b|') + "\\b"

def get_certainty(chat):
    """ Calculates a score of how "certain" a given expression is, using the Certainty Lexicon.

    Source: Rocklage et al. (2023): https://journals.sagepub.com/doi/pdf/10.1177/00222437221134802?casa_token=teghxGBQDHgAAAAA:iby1S-4piT4bQZ6-1lPNGOKUJsx-Ep8DaURu1OGvjuRWDbOf5h6AyfbSLVUgHjyIv31D_aS6PPbT

    Certainty is an individual’s subjective sense of confidence or
    conviction (Petrocelli, Tormala, and Rucker 2007).

    The score is computed by using a regular expression match from the lexicon published in Rocklage et al. (2023);
    the method was developed in consultation with the original author.

    The score ranges from 0 ("very uncertain") to 9 ("very certain").

    If a match is not found, the default value is 4.5 (which we take to be neutral). The default value is the only deviation
    that we make from the main paper, as the original paper simply returns NA.
    
    Args:
        chat (str): The message (utterance) for which we are seeking to evaluate certainty.
    
    Returns:
        float: The certainty score of the utterance.
    """
    
    # default certainty value is 4.5; aka a "neutral" statement in the event we don't find anything
    DEFAULT_CERTAINTY = 4.5

    # pattern match via re library
    certainty_score = 0
    matches = re.findall(master_regex, chat)

    for match in matches:
        certainty_score += certainty.loc[certainty['Word'] == match]["Certainty"].iloc[0]

    # safeguard against division by zero error
    if (len(matches) == 0):
        return DEFAULT_CERTAINTY 
    return (certainty_score / len(matches))