import pandas as pd
import numpy as np
import requests
import sklearn
import nltk
import stanza
from itertools import combinations
from nltk import ngrams
from nltk import jaccard_distance
import torch

# MODELS_DIR = '/Users/priyadcosta/Desktop/Positivity/Random/stanford-corenlp-4.5.2'
# NOTE: when running for the first time, you need to run the following in order to download stanza (for POS tagging)
# stanza.download('en')

def jaccard_distance(set1, set2):
    return len(set1.intersection(set2)) / float(len(set1.union(set2)))

# The function to calculate the Jaccard distance for discourse markers and sets the status accordingly
def discourse_markers_check(df, on_column, threshold, n, discourse_markers):
    
    # Iterate over each row of the dataframe
    for i, row in df.iterrows():
        max_jaccard = 0

        # Tokenize the words and create ngrams
        words = nltk.word_tokenize(row[on_column])
        words_set = nltk.ngrams(words, n)
        
        # Create a list to hold the ngram sets for each row
        ngram_sets = []

        # Iterate over each ngram in the set of ngrams
        for ngram in words_set:
            # Create a set of ngrams, frozenset is used to create an immutable set of each n-gram so that it can be added to the set.
            #if we do not use frozen set, we will not be able to add a set object to a set. Hence, we use frozen set
            ngram = set([frozenset(ngram[j:j+n]) for j in range(len(ngram)-n+1)])
            # Add the set of ngrams to the list
            ngram_sets.append(ngram)

        # Iterate over each set in the list
        for ngram_set in ngram_sets:
            # Calculate the Jaccard distance with the universal set i.e discourse markers
            jaccard = jaccard_distance(set(ngram_set), set(discourse_markers))
            if jaccard > max_jaccard:
                max_jaccard = jaccard
                
        # Store the maximum Jaccard distance for the sets in the list
        df.at[i, 'Max_Jaccard'] = max_jaccard
        
        # Mark the row as a hedged sentence if the maximum Jaccard distance is below the threshold
        if max_jaccard >= threshold: 
            df.at[i, 'DM_output'] = True
        else:
            df.at[i, 'DM_output'] = False


def booster_helper(words_list, booster_words, negative_words=["not", "without"]):

    #iterate throught the list of words in the text and check if a booster word is preceeded by "not" or "without"
    for i in range(1, len(words_list)):
        if words_list[i] in set(booster_words)and words_list[i-1] in negative_words:
          return True
    return False

def booster_check(df, on_column, booster_words, negative_words=["not", "without"]):
    # Apply the check_word function to each row of the DataFrame, after splitting the contents of each data frame into a list
    df['booster_output'] = df[on_column].apply(lambda x: booster_helper(x.split(), booster_words, negative_words=["not", "without"]))


# Load the English language model
nlp = stanza.Pipeline('en')

def is_true_hedged_term(node_tuple,tree,tokens,hedge_words):
    '''   
    Rule 1: If token t is (i) a root word, (ii) has the part-of-speech
    VB* and (iii) has an nsubj (nominal subject) dependency
    with the dependent token being a first person pronoun (i,
    we), t is a hedge, otherwise, it is a non-hedge.
   
    Rule 2: If token t is followed by a token with part-of-speech
    IN (preposition/subordinating conjunction), t is a non-hedge, otherwise, hedge
 
    Rule 3: If token t has a ccomp (clausal complement) dependent, t is a hedge, otherwise, non-hedge.
   
    Rule 4: If token t has an xcomp (open clausal complement)
    dependent d and d has a mark dependent to, t is a nonhedge, otherwise, it is a hedge.

    Rule 5: If token t has an xcomp (open clausal complement)
    dependent, t is a hedge, otherwise, it is a non-hedge.
    
    COMBINATION OF 3,4,5
    Rule 6: If token t has a ccomp (clausal complement) or xcomp -
    (open clausal complement) dependent, t is a hedge, otherwise, it is a non-hedge.

    Rule 7: If token t has relation amod with its head h and h has
    part of speech N (Noun)*, t is a non-hedge, otherwise, it is a hedge.
  
    Rule 8: If token t has relation aux (auxiliary verb) with its head h and h has
    dependent have, t is a non-hedge, otherwise, it is a hedge

    Rule 9: If token t is followed by token than, t is a non-hedge,
    otherwise, it is a hedge.
    '''
    # governor has the following format: (ORIGINAL_NODE, relationship, OBJECT_NODE)
    (focal_node,relation,object_node)  = node_tuple # this is the 'governor' of the word; defines its relationship
   
    # Look for the Hedge Word
    hedge_word_found = None
    # TODO - we might need to stem the text before checking; e.g., "supposed --> suppose"
    # if(stem(focal_node.text) in set(hedge_words) or stem(object_node.text) in set(hedge_words)):
    if(focal_node.text in set(hedge_words) or object_node.text in set(hedge_words)):
        is_hedge_word = True
        if(focal_node.text in set(hedge_words)):
            hedge_word_found = focal_node
        else:
            hedge_word_found = object_node
    # skip if it's not a hedge word
    if(not hedge_word_found): return False

    if focal_node.xpos.startswith('VB') :
        if(focal_node.deprel == 'root' and relation == 'nsubj' and object_node.text.lower() in ['i', 'we']):
            return True
    if(int(hedge_word_found.id) < len(tokens)): # it's not the last word; thus, there is a word afterwards
        if (tokens[int(hedge_word_found.id)].words[0].xpos == 'IN'):
            return False
    if 'ccomp' in relation:
        return True
    if 'xcomp' in relation:
        if object_node.feats == "VerbForm=Inf":
            return False
        else:
            return True
    if 'amod' in relation:
        head = hedge_word_found.head
        head_node = tokens[head-1]
        if head_node.words[0].xpos.startswith('N'):
            return False
        else:
            return True
    if 'aux' in relation:
        # go to the head
        head = hedge_word_found.head # this is a number
        head_node_text = tokens[head-1].text # 1-indexing because 0 in the tree is ROOT; 0 in tokens is 1st token
        have_detected = False
        for node in tree:
            if node[0].text == head_node_text and node[2].text == 'have':
                have_detected = True
                break
            else:
                continue
        return(not have_detected and is_hedge_word)
    if(int(hedge_word_found.id) < len(tokens)): # it's not the last word; thus, there is a word afterwards
        if (tokens[int(hedge_word_found.id)].words[0].text == 'than'):
            return False
    return True

def hedge_check_helper(sentence,hedge_words):

    #creates a parsed sentence. Eg. Priya ate a Pizza will yield an output of [('nsubj', 1, 0), ('ROOT', 0, 2), ('det', 4, 3), ('obj', 2, 4)]
    # ('nsubj', 1, 0) - Priya is the subject of the word "ate"
    # ('ROOT', 0, 2) - "ate is the root of the sentence"
    # ('det', 4, 3) - "a" is the determiner of the pizza
    # ('obj', 2, 4) - Pizza is the object of the verb "ate"

    #creates a tree using the dependencies described above. The root is the root of the tree. We use 0 as we do not know how long the sentence is 
    doc = nlp(sentence)
    tree = doc.sentences[0].dependencies
    tokens = doc.sentences[0].tokens
    # Determine if each value in the tree is a hedge or a non-hedge
    for node in tree:
        # If the node has a valid word in the sentence
        if node[0].text and node[0].text != 'ROOT':
        
            # Check if it is a hedged term as per the 8 rules above and it is present in the list of hedged words
            if is_true_hedged_term(node, tree, tokens, hedge_words):
                return True

    return False       

#applies the function to the dataframe
def hedge_check(df,on_column,hedge_words):
    df['hedge_output'] = df[on_column].apply(lambda x: hedge_check_helper(x,hedge_words))

def is_hedged_sentence2(df,on_column, discourse_markers, booster_words, hedge_words):

    #run discourse marker check
    discourse_markers_check(df, on_column, 0.75, 1, discourse_markers)
    
    #run booster check
    booster_check(df, on_column,booster_words, negative_words=["not", "without"])

    #run hedge check
    hedge_check(df,on_column,set(hedge_words))

    for i, row in df.iterrows():
        if (df.at[i,'booster_output'] or df.at[i,'DM_output'] or df.at[i,'hedge_output']) == True:
            df.at[i, 'hedge_advanced'] = '1'
        else:
            df.at[i, 'hedge_advanced'] = '0'