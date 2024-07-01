import pandas as pd
import numpy as np

from .discursive_diversity import *

"""
Computes the variance in discursive diversity across the chunks for each conversation.

This features takes advantage of the discursive diversity feature, which computes the degree of divergence amongst the meanings conveyed by speakers in a given conversation.

Args:
    chat_data (pd.DataFrame): The chat data, which includes the conversation number, chunk number, and message embeddings.

Returns:
    pd.DataFrame: A grouped DataFrame that contains the conversation identifier as the key, and contains a new column ("variance_in_DD") for each conversation's variance in discursive diversity.
"""
def get_variance_in_DD(chat_data):
    dd_results = chat_data.groupby(['conversation_num', 'chunk_num']).apply(get_DD)
    dd_results = dd_results.reset_index(drop=True)
    results = dd_results.groupby("conversation_num", as_index=False).var()[['conversation_num', 'discursive_diversity']]
    return results.rename(columns={'discursive_diversity': 'variance_in_DD'})
