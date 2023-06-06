import os
import requests
'''
Several functions involve comparing text against a specific lexicon.
In order to prevent repeated re-loading of the lexicon, we pre-load them here.
'''

"""
Returns the list of easy words according to Dale-Chall readability.
"""
def get_dale_chall_easy_words():
    #get the list of dale-chall words
    with open('./features/lexicons/dale_chall.txt', 'r') as file:
        easy_word_list = [line.strip() for line in file]
    return easy_word_list

"""
Returns the list of function words, according to Ranganath, Jurafsky, and McFarland (2013).
https://web.stanford.edu/~jurafsky/pubs/ranganath2013.pdf
"""
def get_function_words():
    #get the list of dale-chall words
    with open('./features/lexicons/function_words.txt', 'r') as file:
        function_word_list = [line.strip() for line in file]
    return function_word_list

"""
Returns a list of question words.
"""
def get_question_words():
    #get the list of question words
    with open('./features/lexicons/question_words.txt', 'r') as file:
        question_word_list = [line.strip() for line in file]
    return question_word_list

"""
Returns a list of discourse marker words.
"""
def get_discourse_markers():
    #get the lists of discourse markers
    dm_path = 'https://raw.githubusercontent.com/hedging-lrec/resources/master/discourse_markers.txt'
    return(requests.get(dm_path).text.strip().split('\n'))

"""
Returns a list of booster words.
"""
def get_booster_words():
    #create a list of booster words
    booster_path = 'https://raw.githubusercontent.com/hedging-lrec/resources/master/booster_words.txt'
    return(requests.get(booster_path).text.strip().split('\n'))

"""
Returns a list of hedge words (used for the advanced, hedge 2.0).
"""
def get_advanced_hedge_words():
    #create a list of hedge words
    hedge_path = 'https://raw.githubusercontent.com/hedging-lrec/resources/master/hedge_words.txt'
    return(requests.get(hedge_path).text.strip().split('\n'))