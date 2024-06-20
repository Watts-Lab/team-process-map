import numpy as np
import pandas as pd

def reduce_chunks(num_rows, max_num_chunks):
    """
    Reduce the number of chunks based on the number of rows.

    This function adjusts the number of chunks to ensure that each chunk has at least two rows.
    If the total number of rows is less than twice the maximum number of chunks, 
    it reduces the number of chunks accordingly.

    :param num_rows: Total number of rows
    :type num_rows: int
    :param max_num_chunks: Initial maximum number of chunks
    :type max_num_chunks: int

    :return: Adjusted number of chunks
    :rtype: int
    """
    if (num_rows < max_num_chunks * 2):
        max_num_chunks = int(num_rows / 2)
    if max_num_chunks < 1:
        return 1
    else:
        return max_num_chunks
    
def create_chunks_messages(chat_data, num_chunks):
    """
    Assign chunk numbers to the chats within each conversation based on the number of messages.

    This function ensures that there is an even number of messages per chunk by calculating 
    the chunk size for each conversation and adjusting the chunk number accordingly.

    :param chat_data: Dataframe containing chat data
    :type chat_data: pd.DataFrame
    :param num_chunks: Initial maximum number of chunks
    :type num_chunks: int

    :return: Dataframe with an additional 'chunk_num' column indicating chunk assignments
    :rtype: pd.DataFrame
    """

    # Calculate the total number of rows per conversation
    conversation_lengths = chat_data.groupby('conversation_num').size()

    chunks = conversation_lengths.apply(lambda x: reduce_chunks(x, num_chunks))

    # Calculate the chunk size based on the total number of conversations
    chunk_size = np.ceil(conversation_lengths / chunks) 
    
    for i, group in chat_data.groupby('conversation_num'): # for each group
        chunk_num = 0
        counter = 0

        for chat_id in group.index.values: # iterate over the index values
            chat_data.at[chat_id, 'chunk_num'] = str(chunk_num)

            counter += 1

            #if counter = 1 for the last row of a group (implies last chunk has one element), and the chunk num > 0, then just make the last one - 1
            if counter == 1 and chunk_num > 0 and chat_id == group.index.values[-1]:
                chat_data.at[chat_id, 'chunk_num'] = str(chunk_num - 1)

            if counter == chunk_size[i] and chunk_num < chunks[i] - 1: # assign any extras to the last chunk
                chunk_num += 1
                counter = 0    

    return(chat_data)

def create_chunks(df, num_chunks):
    """
    Assign chunk numbers to the chats within each conversation based on time.

    This function divides each conversation into time-based chunks, ensuring each chunk spans an equal duration.

    :param df: DataFrame containing chat data with a 'timestamp' column
    :type df: pd.DataFrame
    :param num_chunks: Number of chunks to divide the conversation into
    :type num_chunks: int

    :return: DataFrame with an additional 'chunk_num' column indicating time-based chunk assignments
    :rtype: pd.DataFrame
    """

    final_df = pd.DataFrame(columns=df.columns)

    # Replace instances of NULL_TIME; this throws off the type checking
    df['timestamp'] = df['timestamp'].replace('NULL_TIME', None)
    timestamps = df['timestamp'].dropna()

    is_datetime_string = False

    # Check the type of the timestamp string
    if (isinstance(timestamps[0], str)): # DateTime String, e.g., '2023-02-20 09:00:00'
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        is_datetime_string = True
    elif(isinstance(timestamps[0], int)):
        if(timestamps[0] > 423705600): # this is Unix time; the magic number is a time in 1983!
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        # If it's not Unix time, we can treat it as an int offset

    # Group and calculate difference
    for conversation_num, group in df.groupby('conversation_num'):

        if pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            total_duration_seconds = (group['timestamp'].max() - group['timestamp'].min()).total_seconds() if is_datetime_string else (group['timestamp'].max() - group['timestamp'].min()).total_seconds()
        else:
            total_duration_seconds = (group['timestamp'].max() - group['timestamp'].min())

        # Add a new column for chunk number
        group['chunk_num'] = -1 

        # Catch NA's from the case where people didn't chat or only 1 chat exists --- all chunk nums should be 0
        if pd.isna(total_duration_seconds) or total_duration_seconds == 0:
            group['chunk_num'] = 0

        # Case where people did chat
        else:
            # Calculate the duration of each chunk
            chunk_duration = total_duration_seconds / num_chunks

            # Initialize the 'chunk_num' column
            group['chunk_num'] = -1

            for index, row in group.iterrows():
                # Get the timestamp
                timestamp = row['timestamp']
                # Calculate the chunk number
                if pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                    chunk_number = int(((timestamp - group['timestamp'].min()).total_seconds()) / chunk_duration)
                else:
                    chunk_number = int(((timestamp - group['timestamp'].min())) / chunk_duration)

                # Assign the chunk number for each row
                group.loc[index, 'chunk_num'] = chunk_number

            # restrict the range of the chunks from 0 to num_chunks - 1
            group['chunk_num'] = group['chunk_num'].clip(0, num_chunks - 1)

        final_df = pd.concat([final_df, group], ignore_index=True)

    # Cast the chunk to a string
    final_df['chunk_num'] = str(final_df['chunk_num'])
    return final_df


def assign_chunk_nums(chat_data, num_chunks, use_time_if_possible = True):
    """
    Assign chunks to the chat data, splitting it into "equal" pieces.

    This functionality is necessary for some conversational features that examine what happens throughout the course
    of a conversation (e.g., in the beginning, middle, and end).

    This function has slightly different behavior depending on whether timestamps are available, and depending on user speciifcations.

    If a timestamp column exists and `use_time_if_possible` is True, the function will chunk based on the timestamp.
    Otherwise, it will chunk based on the number of messages.

    :param chat_data: The input chat data
    :type chat_data: pd.DataFrame
    :param num_chunks: The number of chunks desired
    :type num_chunks: int
    :param use_time_if_possible: If a timestamp exists, chunk based on the timestamp rather than based on the number of messages. Defaults to True.
    :type use_time_if_possible: bool, optional

    :return: DataFrame with chunk numbers assigned
    :rtype: pd.DataFrame
    """
    if 'timestamp' in chat_data.columns and use_time_if_possible:
        return create_chunks(chat_data, num_chunks)
    else:
        return create_chunks_messages(chat_data, num_chunks)