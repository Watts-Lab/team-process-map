import re

from features.basic_features import *

NTRI_regex = "what\?+|sorry|excuse me|huh\??|who\?+|pardon\?+|say.*again\??|what'?s that|what is that"
def classify_NTRI(text):
  """
  Classify whether the message contains clarification questions, such as "what?" "sorry?" etc.

  Performs a simple regex matching over a series of repair indicators from Ranganath et al. (2013).
  Source: https://sites.socsci.uci.edu/~lpearl/courses/readings/RanganathEtAl2013_DetectingFlirting.pdf

  Args:
    text (str): The message (utterance) being analyzed.

  Returns:
    int: The number of matches for repair indicators.

  """
  return 1 if re.match(NTRI_regex, text) else 0
  
## Calculate the word type-to-token ratio
def get_word_TTR(text):
  """
  Get the word type-token ratio, calculated as follows:

  Number of Unique Words / Number of Total Words.

  The function assumes that punctuation is retained when being inputted, but parses it out within the function.

  Args:
    text (str): The message (utterance) being analyzed.
  
  Returns:
    float: The word type-token ratio.

  """
  # remove punctuations
  new_text = re.sub(r"[^a-zA-Z0-9 ]+", '',text)
  # calculate the number of unique words
  num_unique_words = len(set(new_text.split()))
  # calculate the word type-to-token ratio
  if count_words(new_text) == 0:
    return 0
  else:
    return num_unique_words/count_words(new_text)   

def get_proportion_first_pronouns(df):
  """
  Get the proportion of first person pronouns: the total number of first person words divided by the total number of words.

  Note that the function assumes that lexical features and basic features have already been run, and have generated a raw count of
  first-person pronouns (stored in "first_person_raw") and a total number of words (stored in "num_words").

  Args:
    df (pd.DataFrame):  This is a pandas dataframe of the chat level features.

  Returns:
    pd.Series: A column in the chat-level dataframe, in which we calculate the number of first-person pronouns over the total number of words. Defaults to 0 in case of a DIV/0 error.

  """
  return(df.apply(lambda row: row["first_person_raw"] / row["num_words"] if row["num_words"] > 0 else 0, axis=1))