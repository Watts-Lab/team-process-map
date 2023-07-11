import pandas as pd
import numpy as np
from .discursive_diversity import get_cosine_similarity

'''
This is a conversation level feature, which computes the semantic modulation that individuals experience with respect to themselves across each chunk transition. Incongruent modulation measures the variance of the rates of shifting, while within person discursive range measures the average amount of shifting. 

'''

def get_nan_vector():
    f = open("data/vectors/nan_vector.txt", "r")
    str_vec = f.read()
    nan_vector_list = [float(e) for e in str_vec[1:-1].split(',')]
    return np.array(nan_vector_list)


def get_within_person_disc_range(chat_data, num_chunks):

    # Get nan vector 
    nan_vector = get_nan_vector()

    #calculate mean vector per speaker per chunk
    mean_vec_speaker_chunks = pd.DataFrame(chat_data.groupby(['conversation_num', 'speaker_nickname', 'chunk_num']).message_embedding.apply(np.mean)).unstack('chunk_num').rename(columns={'message_embedding': 'mean_chunk_vec'})

    #collapse multi-index
    mean_vec_speaker_chunks.columns = ["_c".join(col).strip() for col in mean_vec_speaker_chunks.columns.values]

    actual_num_chunks = len(mean_vec_speaker_chunks[2:].columns) # omit the first two, which is conversation_num and speaker_nickname

    #each element in inter_chunk_range is a list of cosine distances BETWEEN each pair of consecutive chunks
    inter_chunk_range = [ [] for i in range(actual_num_chunks - 1)]

    for conv_idx, row in mean_vec_speaker_chunks.iterrows():
        for index in range(actual_num_chunks - 1):
            tpl = [row['mean_chunk_vec_c' + str(index)], row['mean_chunk_vec_c'+ str(index+1)]]
            value = 0
            if (pd.isnull(tpl)).all():
                value = np.nan
            elif (pd.isnull(tpl)).any():
                # Compare with Nanvector 
                if type(tpl[0]) == float:     
                    tpl = [nan_vector, tpl[1]]
                else:
                    tpl = [nan_vector, tpl[0]]
                value = 1 - get_cosine_similarity(tpl)[0]
            else:
                value = 1 - get_cosine_similarity(tpl)[0]
            inter_chunk_range[index].append(value)

    index = []
    for i in range(actual_num_chunks - 1):
        index.append("c" + str(i) + "_c" + str(i + 1))
    range_df = pd.DataFrame(inter_chunk_range, index=index).T
    range_df['conversation_num'] = mean_vec_speaker_chunks.reset_index()['conversation_num']
    range_df = range_df.set_index('conversation_num')

    # variance within person discursive range 
    var_disc_range = range_df.groupby('conversation_num').apply(lambda x: np.nanvar(x, axis=0).sum()).to_frame().rename(columns={0:'incongruent_modulation'})
    
    # average within person discursive range 
    avg_disc_range = range_df.groupby('conversation_num').apply(lambda x: np.nanmean(x, axis=0).sum()).to_frame().rename(columns={0:'within_person_disc_range'})

    return pd.merge(
                left=var_disc_range,
                right=avg_disc_range,
                on=['conversation_num'],
                how="inner"
            )
