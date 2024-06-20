import pandas as pd
import re

#returns the total number of hedge words present in the data set (We can add more hedge words)
def is_hedged_sentence_1(num_hedge_words):
    """
    Computes a binary value of whether a sentence contains a hedge words from the list.

    Example hedge words include "sort of," "kind of, "I guess," etc.

    The list is part of the lexicons that are processed in lexical_features_v2, so one dependency of this function is that
    we assume that we have already run the basic lexical features.

    Args:
        num_hedge_words (float): The value of the lexical output for hedge words from running liwc_features() (under lexical_features_v2)

    Returns:
        int: 1 if there is at least one hedge word; 0 otherwise

    """
    return 1 if num_hedge_words > 0 else 0