"""
file: lexical_features_v2.py
---
A faster version of the lexical_features.py file.
"""
# Importing packages
import pickle
import re
import pandas as pd
import os
from pathlib import Path

# Store the LIWC features as a RATE per 100 words, per best practice (cite: https://www.mikeyeomans.info/papers/PGCR_yeomans.pdf, p. 42)
# len(re.findall(regex, chat)) is the raw count
# Formula: count / chat length * (chat length / 100) -> count / 100
def get_liwc_rate(regex, chat):
	if(len(chat) > 0):
		return (len(re.findall(regex, chat))/(len(chat)))*(len(chat)/100)
	else:
		return 0

def liwc_features(chat_df: pd.DataFrame) -> pd.DataFrame:
	"""
		This function takes in the chat level input dataframe and computes the lexicon feautres for the 'message' column.
		TODO: The current implementation assumes presence of 'message' column. 
			  Might need to abstract this in order to make this generalizable for all datasets.
			  
	PARAMETERS:
		@param chat_df (pd.DataFrame): This is a pandas dataframe of the chat level features. Should contain 'message' column.

	RETURNS:
		(pd.DataFrame): Dataframe of the lexical features stacked as columns.
	"""
	# Load the preprocessed lexical regular expressions
	current_script_directory = Path(__file__).resolve().parent
	lexicon_pkl_file_path = current_script_directory / "lexicons_dict.pkl"
	with open(lexicon_pkl_file_path, "rb") as lexicons_pickle_file:
		lexicons_dict = pickle.load(lexicons_pickle_file)
	
	# Return the lexical features stacked as columns
	return pd.concat(
		# Finding the # of occurrences of lexicons of each type for all the messages.
		[pd.DataFrame(chat_df["message_original"].apply(lambda chat: get_liwc_rate(regex, chat)))\
										.rename({"message_original": lexicon_type + "_lexical_per_100"}, axis=1)\
			for lexicon_type, regex in lexicons_dict.items()], 
		axis=1
	)
