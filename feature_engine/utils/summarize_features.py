import numpy as np

'''
These functions are for the purpose of calculating aggregates of various user level features at the conversation level.

'''

'''
function: get_user_sum_dataframe

Returns a dataframe that gets the total number of X per individual, where X could be # words, etc.

In other words, this dataframe first groups by the user and SUMS all of a feature, and returns the sum.

@param chat_level_data = a dataframe in which each row is one chat.
@param on_column = the name of the numeric column, X, which is summed per individual
'''
def get_user_sum_dataframe(chat_level_data, on_column, speaker_id = "speaker_nickname"):
    grouped_conversation_data = chat_level_data[["conversation_num", speaker_id, on_column]].groupby(["conversation_num", speaker_id]).sum().reset_index()
    grouped_conversation_data = grouped_conversation_data.rename(columns = {on_column: "sum_"+on_column})
    # gets this dataframe:
    # Batch# Round# Speaker  Total Number of Words
    # 0      1      Priya    100
    # 0      1      Yuluan   90
    return(grouped_conversation_data)


'''
function: get_user_average_dataframe

Returns a dataframe that gets the average number of X per individual, where X could be # words, etc.

In other words, this dataframe first groups by the user and AVERAGES all of a feature, and returns the average.

@param chat_level_data = a dataframe in which each row is one chat.
@param on_column = the name of the numeric column, X, which is summed per individual
'''
def get_user_average_dataframe(chat_level_data, on_column, speaker_id = "speaker_nickname"):
    grouped_conversation_data = chat_level_data[["conversation_num", speaker_id, on_column]].groupby(["conversation_num", speaker_id]).mean().reset_index()
    grouped_conversation_data = grouped_conversation_data.rename(columns = {on_column: "average_"+on_column})    # gets this dataframe:
    # Batch# Round# Speaker  Average Number of Words
    # 0      1      Priya    100
    # 0      1      Yuluan   90
    return(grouped_conversation_data)


'''
function: get_average()

Returns: grouped dataframe; [conversation_num, average_of_input_column]

@param input_data = df of data at the chat/user level
@param column_to_summarize = the column being aggregated
@param new_column_name = the desired name of the new summary column
'''
def get_average(input_data, column_to_summarize, new_column_name):
    # grouped_conversation_data = get_count_dataframe(chat_level_data, column_to_summarize)
    input_data[new_column_name] = input_data.groupby(["conversation_num"], sort=False)[column_to_summarize].transform(lambda x: np.mean(x))
    return(input_data[["conversation_num", new_column_name]].drop_duplicates())

'''
function: get_max()

Returns: grouped dataframe; [conversation_num, max_of_input_column]

@param input_data = df of data at the chat/user level
@param column_to_summarize = the column being aggregated
@param new_column_name
'''
def get_max(input_data, column_to_summarize, new_column_name):
    # grouped_conversation_data = get_count_dataframe(chat_level_data, column_to_summarize)
    input_data[new_column_name] = input_data.groupby(["conversation_num"], sort=False)[column_to_summarize].transform(max)
    return(input_data[["conversation_num", new_column_name]].drop_duplicates())

'''
function: get_min()

Returns: grouped dataframe;[conversation_num, min_of_input_column]

@param input_data = df of data at the chat/user level
@param column_to_summarize = the column being aggregated
@param new_column_name
'''
def get_min(input_data, column_to_summarize, new_column_name):
    # grouped_conversation_data = get_count_dataframe(chat_level_data, column_to_summarize)
    input_data[new_column_name] = input_data.groupby(["conversation_num"], sort=False)[column_to_summarize].transform(min)
    return(input_data[["conversation_num", new_column_name]].drop_duplicates())

'''
function: get_stdev()

Returns: grouped dataframe; [conversation_num, stdev_of_input_column]

@param input_data = df of data at the chat/user level
@param column_to_summarize = the column being aggregated
@param new_column_name
'''
def get_stdev(input_data, column_to_summarize, new_column_name):
    # grouped_conversation_data = get_count_dataframe(chat_level_data, column_to_summarize)
    input_data[new_column_name] = input_data.groupby(["conversation_num"], sort=False)[column_to_summarize].transform(lambda x: np.std(x))
    return(input_data[["conversation_num", new_column_name]].drop_duplicates())

'''
function: get_sum()

Returns: grouped dataframe; [conversation_num, stdev_of_input_column]

@param input_data = df of data at the chat/user level
@param column_to_summarize = the column being aggregated
@param new_column_name
'''
def get_sum(input_data, column_to_summarize, new_column_name):
    # grouped_conversation_data = get_count_dataframe(chat_level_data, column_to_summarize)
    input_data[new_column_name] = input_data.groupby(["conversation_num"], sort=False)[column_to_summarize].transform(lambda x: np.sum(x))
    return(input_data[["conversation_num", new_column_name]].drop_duplicates())

