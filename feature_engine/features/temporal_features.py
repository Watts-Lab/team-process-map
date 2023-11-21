import pandas as pd
import numpy as np

def get_time_diff(df,on_column):
    # Replace instances of NULL_TIME
    df[on_column] = df[on_column].replace('NULL_TIME', None)

    #convert timestamp column to datetime type (in minutes)
    if(isinstance(df[on_column][0], str)): # String datetime, e.g., '2023-02-20 09:00:00'
        df[on_column] = pd.to_datetime(df[on_column])
    elif(isinstance(df[on_column][0], np.int64)): # Int Unix dateime, e.g., '1677262112288'
        df[on_column] = pd.to_datetime(df[on_column], unit='ms')

    # set and zero time_diff column
    df["time_diff"] = np.zeros(len(df))

    for i in range(1, len(df)):
        if df.loc[i, "conversation_num"] == df.loc[i-1, "conversation_num"]: # only do this if they're in the same conversation
            df.loc[i, "time_diff"] = (df.loc[i, on_column] - df.loc[(i-1), on_column]) / pd.Timedelta(seconds=1)

    return df['time_diff']

def get_time_diff_startend(df):
    # set and zero time_diff column
    df["time_diff"] = np.zeros(len(df))

    for i in range(1, len(df)):
        if df.loc[i, "conversation_num"] == df.loc[i-1, "conversation_num"]: # only do this if they're in the same conversation
            df.loc[i, "time_diff"] = (df.loc[i, "timestamp_start"] - df.loc[(i-1), "timestamp_end"])

    return df['time_diff']