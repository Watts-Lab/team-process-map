import pandas as pd

"""
Conversation Level Feature:

Feature that returns the difference of overall positivity - negativity at the conversation level to assess how potential conflict affects the group discussion, as well as the general balance between positivity and negativity.

"""


# def group_affective_balance(chat_df):

#     gab = []
#     for conv in chat_df.groupby(['conversation_num']):
#         # print(conv[1]['positive_bert'])

#         gab.append(conv[1]['positive_bert'].sum() - conv[1]['negative_bert'].sum())
    
#     return pd.DataFrame({'gab':gab})

def group_affective_balance(chat):

    gab = []
    for conv in chat_df.groupby(['conversation_num']):
        # print(conv[1]['positive_bert'])

        gab.append(conv[1]['positive_bert'].sum() - conv[1]['negative_bert'].sum())
    
    return pd.DataFrame({'gab':gab})
