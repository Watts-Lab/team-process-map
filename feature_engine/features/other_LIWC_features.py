from collections import Counter
import re

from features.basic_features import *



## Get the number of question marks in one message
def num_question(text):
  return Counter(text)["?"]


## Classify whether the message contains clarification questions
NTRI_list = ("what?","sorry","excuse me","huh?","who?","pardon?","say again?","say it again?","what's that","what is that")
def NTRI(text):
  if len([x for x in NTRI_list if x in text]) > 0:
    return 1
  else:
    return 0
  

## Calculate the word type-to-token ration
def word_TTR(text):
  # remove punctuations
  new_text = re.sub(r"[^a-zA-Z0-9 ]+", '',text)
  # calculate the number of unique words
  num_unique_words = len(set(new_text.split()))
  # calculate the word type-to-token ratio
  return num_unique_words/count_words(new_text)   


## Proportion of first person pronouns
first_pronouns = ["i",'me','mine','myself','my','we','our','ours','ourselves','lets']
def proportion_first_pronouns(text):
  num_first_prononouns = len([x for x in text.split() if x in first_pronouns])
  return num_first_prononouns/count_words(text)
