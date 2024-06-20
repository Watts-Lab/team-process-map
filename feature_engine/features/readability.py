import pandas as pd
import re
import nltk
import pyphen

from features.basic_features import count_words

# Define the function to calculate the Dale-Chall score
def count_syllables(word):
    """
    Count the number of syllables in a word.
    
    Args:
        word(str): The input word.

    Returns:
        int: The number of syllables in the word.
    """
    dic = pyphen.Pyphen(lang='en')
    pyphen_result = dic.inserted(word)
    return len(re.findall(r"-", pyphen_result))

def count_difficult_words(text, easy_words):
    """
    Count the number of difficult words in a text. The difficult words are those that are not in
    an "easy words" list (passed in from the ChatLevelFeaturesCalculator, and originating in the get_dale_chall_easy_words() in Utilities).

    Args:
        text(str): The message (utterance) being analyzed.
        easy_words(list): A list of "easy" words according to Dale-Chall. This comes from the Utilities.

    Returns:
        The number of difficult words.
    """
    
    difficult_words = 0
    # recall that words are already pre-processed; substitute punctuation here for words only
    words = re.sub(r"[^a-zA-Z0-9 ]+", '',text).split()

    remaining_words = [i for i in words if not i in easy_words]

    for word in remaining_words:
        # words with more than 3 syllables are difficult
        if(count_syllables(word) >= 3):
            difficult_words += 1
    
    return difficult_words

def dale_chall_helper(text, **easy_words):
    """
     Calculate the Dale-Chall readability score of a text. The Dale-Chall score are defined as:

        0.1579 * ((difficult_words / words) * 100) + 0.0496 * (words / sentences)

        If the percentage of difficult words is above 5%, then add 3.6365 to the raw score to get the adjusted score, otherwise the adjusted score is equal to the raw score.

     In general, lower scores mean that a text is easier to read, and higher scores indicate that a text is harder to read.

     Because of the need to split up sentences, this function requires the version of pre-processed text WITH punctuation retained.

     Source: https://en.wikipedia.org/wiki/Dale%E2%80%93Chall_readability_formula
     Citation (of example usage) in Cao et al. (2020): https://dl.acm.org/doi/pdf/10.1145/3432929 

     Args:
        text(str): The message (utterance) being analyzed.
        easy_words(list): A list of "easy" words according to Dale-Chall. This comes from the Utilities.

     Returns:
        float: The Dale-Chall Readability Score.

    """

    num_words = count_words(text)
    num_sentences = len(re.split(r'[.?!]\s*', text)) 
    avg_sentence_length = num_words/num_sentences
    num_difficult_words = count_difficult_words(text, easy_words)

    #get the percentage of difficult words(odw)
    if num_words == 0:
        pdw = 0
    else:
        pdw = num_difficult_words/num_words*100
        
    # magic numbers are due to Dale-Chall formula
    raw_score = (0.1579*pdw) + (0.0496*avg_sentence_length)
    if pdw > 5:
        raw_score += 3.6365
    return raw_score

def classify_text_dalechall(score):
    """
    Classifies the Dale-Chall score into a category of "easy," "medium," or "difficult":

    - A score is easy if it is below 4.9 (readable by a 4th grader or below)
    - A score is medium if is is between 4.9 and 5.9 (readable by a middle school student)
    - A score is difficult if it is above 5.9

    Args:
        score: the Dale-Chall readability score.

    Returns:
        str: The label/classification associated with the text.
    """
    if score <= 4.9:
        return "easy"
    elif score <= 5.9:
        return "medium"
    else:
        return "difficult"
