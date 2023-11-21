import pandas as pd
import numpy as np

'''
User Level Feature.

This feature takes in data at the user level, and generates a "user_list" per user per conversation. This "user_list" contains the other participating in this conversation. 

'''

def remove_active_user(df):
    counter = 0
    for index, row in df.iterrows():
        user = row['speaker_nickname']
        user_list = row['user_list']
        i = np.argwhere(user_list==user)
        df.at[index,'user_list'] = np.delete(row['user_list'],i)
        df.iloc
    return df

def get_user_network(user_df):
    
    user_lists = user_df.groupby(["conversation_num"]).apply(lambda df : np.asarray(df["speaker_nickname"]))

    user_list_df = pd.merge(
        left=user_df,
        right=user_lists.rename("user_list"),
        on=['conversation_num'],
        how="inner"
    )
    
    user_list_df_final = user_list_df.groupby(["conversation_num"]).apply(lambda df : remove_active_user(df)).reset_index(drop=True)
    
    return user_list_df_final[['conversation_num', 'speaker_nickname', 'user_list']]