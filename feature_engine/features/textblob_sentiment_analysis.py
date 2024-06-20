import pandas as pd
from textblob import TextBlob
import statistics as stat

def get_subjectivity_score(string):
    """
    Uses the TextBlob pacakge to obtain the "subjectivity" score of a text.

    The score ranges between 0 to 1, 0 - purely factual, 1 - pure opinion.

    We point you to the TextBlob reference for more information: https://textblob.readthedocs.io/en/dev/api_reference.html#textblob.blob.TextBlob.sentiment

    Citation (of example usage) in Cao et al. (2020): https://dl.acm.org/doi/pdf/10.1145/3432929 

    Args:
        string(str): The message (utterance) being analyzed.

    Returns:
        float: The subjectivity score, in the range [0.0, 1.0]

    """
    return TextBlob(string).sentiment.subjectivity
def get_polarity_score(string):
    """
    Uses the TextBlob pacakge to obtain the "polarity" score of a text.

    The score ranges between -1 to 1,  -1 is most negative, 0 is neutral, 1 is positive.

    We point you to the TextBlob reference for more information: https://textblob.readthedocs.io/en/dev/api_reference.html#textblob.blob.TextBlob.sentiment

    Citation (of example usage) in Cao et al. (2020): https://dl.acm.org/doi/pdf/10.1145/3432929 

    Args:
        string(str): The message (utterance) being analyzed.

    Returns:
        float: The polarity score, in the range [-1.0, 1.0]

    """
    return TextBlob(string).sentiment.polarity