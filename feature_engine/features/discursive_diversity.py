import pandas as pd
import numpy as np
import itertools
from sklearn.metrics.pairwise import cosine_similarity

'''
This is a conversation level feature, which computes degree of divergence amongst the meanings conveyed by speakers in a given conversation. 

'''

# From given code: https://pubsonline.informs.org/doi/suppl/10.1287/mnsc.2021.4274
def get_unique_pairwise_combos(lst):
    '''Computes all unique pairwise combinations of the elements in a list.
    input: array or list
    output: list of unique pairwise combinations of elements of the input list'''
    return list(itertools.combinations(lst, 2))


def get_cosine_similarity(vecs):
    '''computes cosine similarity between a list of vectors
    input: list of vectors (vecs) (this has to be a pair!)
    output: cosine similarity
    '''
    if len(vecs) > 1:
        cos_sim_matrix = cosine_similarity(vecs)
        return cos_sim_matrix[np.triu_indices(len(cos_sim_matrix), k = 1)]
    else:
        return np.nan


def get_DD(chat_data):
    
    # Get mean embedding per speaker per conversation
    user_centroid_per_conv = pd.DataFrame(chat_data.groupby(['conversation_num','speaker_nickname'])['message_embedding'].apply(np.mean)).reset_index().rename(columns={'message_embedding':'mean_embedding'})

    # For each team(conversation) get all unique pairwise combinations of members' means:
    user_pairs = pd.DataFrame(user_centroid_per_conv.groupby(['conversation_num'])['mean_embedding'].\
    apply(get_unique_pairwise_combos)).reset_index().\
    rename(columns={'mean_embedding':'user_pairs_per_conv'})

    # Get cosine distances between each pair for every conversation, average all the distances to get DD per conversation
    cos_dists_mean_widay_btwu = []

    for lst in user_pairs.user_pairs_per_conv:

        # Make sure list isn't empty:
        if lst:
            # Store the cosine distances for the person's list of tuples
            cos_dists = []
            for tpl in lst:
                try:
                    cos_d = 1 - get_cosine_similarity([tpl[0], tpl[1]])
                    cos_dists.append(cos_d)
                except ValueError as e:
                    # Occurs when np.nan in tuple
                    pass

            # Compute mean of cos dists
            cos_dists_mean_widay_btwu.append(np.nanmean(cos_dists, dtype="float64"))
        else:
            cos_dists_mean_widay_btwu.append(np.nan)

    user_pairs['discursive_diversity'] =  cos_dists_mean_widay_btwu

    return user_pairs[['conversation_num', 'discursive_diversity']]

