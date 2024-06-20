import pandas as pd
import numpy as np

from features.temporal_features import coerce_to_date_or_number

def burstiness(df, timediff):
    """ Computes the level of "burstiness" in a conversation, or the extent to which messages in a 
    conversation occur periodically (e.g., every X seconds), versus in a "bursty" pattern 
    (e.g., with long pauses and many messages in rapid succession.)

    The coefficient of variation, B, is sourced from Reidl and Wooley (2016): https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2384068

    B = (standard deviation of wait times - mean of wait times) / (standard deviation of wait times + mean of wait times)

    
    Args:
        df (pd.DataFrame): The input dataframe, grouped by the conversation index, to which this function is being applied.
        timediff (str): The column name associated with the time differences between messages in a conversation (computed in a pre-processing step.) 
    
    Returns:
        float: The team burstiness score (B)
    """

    if timediff not in df.columns:
        return None 
    
    # Check for any NA values and drop them accordingly
    df[timediff] = df[timediff].apply(coerce_to_date_or_number)

    wait_times = (df[timediff].dropna()).astype(float).values

    if len(wait_times) <= 1:
        return 0
    
    # Compute coefficient of variation measure B (Goh & Barabasi 2008)
    standard_deviation = np.std(wait_times)
    mean = np.mean(wait_times)
    B = (standard_deviation - mean) / (standard_deviation + mean)
    return B

def get_team_burstiness(df, timediff):
    """ Applies the burstiness coefficient to each conversation in the utterance (chat)-level dataframe and returns a conversation-level dataframe.
    The Burstiness feature takes advantage of the fact that we already compute the time difference between messages
    as one of the utterance (chat)-level features.

    Args:
        df (pd.DataFrame): The utterance (chat)-level dataframe.
        timediff (str): The column name associated with the time differences between messages in a conversation (computed by the utterance-level feature, get_temporal_features.) 

    Returns:
        pd.DataFrame: a grouped dataframe that contains the conversation identifier as the key, and contains a new column ("team_burstiness") for each group's burstiness coefficient.
    """
    if timediff not in df.columns:
        print(f"Temporal Features are nonexistent for this dataset.")
        return None
    
    # Applies burstiness function to overall dataframe and then groups coefficient by conversation number
    burstiness_coeff = df.groupby("conversation_num").apply(lambda x : burstiness(x, timediff)).reset_index().rename(columns={0: "team_burstiness"})
    return burstiness_coeff
