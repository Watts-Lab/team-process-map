from .get_all_DD_features import *
import pandas as pd
import numpy as np

def get_user_centroids(chat_data, vect_data):

    chats = chat_data.copy()

    # Format data
    chats['message_embedding'] = conv_to_float_arr(vect_data['message_embedding'].to_frame())

    user_centroid_per_conv = pd.DataFrame(chats.groupby(['conversation_num','speaker_nickname'])['message_embedding'].apply(np.mean)).reset_index().rename(columns={'message_embedding':'mean_embedding'})

    return user_centroid_per_conv['mean_embedding']


