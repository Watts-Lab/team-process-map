'''
Instructions:
1. Go to this path https://github.com/bbevis/SECR and dowload the folder named SECR
2. Save the SECR Folder within feature_engine/features
'''


import SECR.System.feature_extraction as fe
import SECR.System.keywords as keywords
import pandas as pd

kw = keywords.kw

def get_politeness_v2(df,on_column):
    """
    @Args:
        The text dataframe
    @Returns:
        The dataframe after adding the politness v2 features
    """

    #extract the column headers by running the script for a random text
    column_headers = fe.feat_counts("hahaha",kw).sort_values(by='Features')['Features'].tolist()
    
    # Apply the function to each row in 'text_column' and store the result in a new column 'output_column'
    df_output = df[on_column].apply(lambda x: fe.feat_counts(x,kw).sort_values(by='Features')['Counts'])

    #add the column headers
    df_output.columns = column_headers

    return df_output