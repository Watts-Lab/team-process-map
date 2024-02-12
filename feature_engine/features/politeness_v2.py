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

    #extract the column headers by running the script for a random text. We sort the names of the features in alphabetical order. 
    #This is done because the original package sorts the features by counts. It is not possible do so if we have a number of rows, as each row may have different counts for different features 
    column_headers = fe.feat_counts("hahaha",kw).sort_values(by='Features')['Features'].tolist()
    
    # Apply the function to each row in 'text_column' and store the result in a new column 'output_column'. We sort the names of the features in alphabetical order
    df_output = df[on_column].apply(lambda x: fe.feat_counts(x,kw).sort_values(by='Features')['Counts'])

    #add the column headers
    df_output.columns = column_headers

    return df_output
