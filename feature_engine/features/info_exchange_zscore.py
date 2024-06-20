import numpy as np
import pandas as pd
import re

def get_info_exchange_wordcount(df, first_person):
  '''
  This functinon computes the total word count in a message minus first person singular pronouns.

  Note that the function assumes that lexical features and basic features have already been run, and have generated a raw count of
  first-person pronouns (stored in "first_person_raw") and a total number of words (stored in "num_words").

  This value then serves as an input into a Z-score (calculated using a function in Utilities); the idea
  is to compute the extent to which a team exchanges more or fewer "content words" outside of first-person pronouns,
  which is a rough measure of "information exchange."

  Because the original paper was not specific about what the z-scores were computed with respect to, we compute
  one version with respect to each chat in a single conversation, and one version with respect to all chats in
  the entire dataset of all conversations.

  Source: Tausczik and Pennebaker (2013); https://www.cs.cmu.edu/~ylataus/files/TausczikPennebaker2013.pdf
  
  Args:
    df (pd.DataFrame):  This is a pandas dataframe of the chat level features.
    first_person (list): A list of first person words. This comes from get_first_person_words() under the Utilities.

  Returns:
    pd.Series: A column containing the difference in total words and first-person singular pronouns.

  '''
  first_person_regex = " | ".join(first_person)
  df['first_person_raw'] = df['message'].apply(lambda chat: len(re.findall(first_person_regex, chat)))
  return (df["num_words"] - df["first_person_raw"])