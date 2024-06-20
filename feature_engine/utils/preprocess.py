import re
import pandas as pd

def preprocess_conversation_columns(df, conversation_id = None, cumulative_grouping = False, within_task = False):
	"""
    Preprocesses conversation data by removing special characters from column names and assigning a conversation number.

    :param df: The DataFrame containing conversation data.
    :type df: pd.DataFrame
    :param conversation_id: The column name to use for assigning conversation numbers. Must be one of {'gameId', 'roundId', 'stageId'}.
    :type conversation_id: str, optional
    :param cumulative_grouping: Whether to group data cumulatively based on the conversation_id.
    :type cumulative_grouping: bool, optional
    :param within_task: Used only if cumulative_grouping is True, to specify if grouping is within the task.
    :type within_task: bool, optional
    :return: The preprocessed DataFrame with a conversation number column.
    :rtype: pd.DataFrame
    """

	# remove all special characters from df
	df.columns = df.columns.str.replace('[^A-Za-z0-9_]', '', regex=True)
	
	# If data is grouped by batch/round, add a conversation num
	if {'batch_num', 'round_num'}.issubset(df.columns):
		df['conversation_num'] = df.groupby(['batch_num', 'round_num']).ngroup()
		df = df[df.columns.tolist()[-1:] + df.columns.tolist()[0:-1]] # make the new column first
	if ({'gameId', 'roundId', 'stageId'}.issubset(df.columns) and conversation_id in {'gameId', 'roundId', 'stageId'}):
		if(cumulative_grouping):
			df = create_cumulative_rows(df, conversation_id, within_task)
			df['conversation_num'] = df['cumulative_Id'] # set it to be the cumulative grouping
		else:
			df['conversation_num'] = df[conversation_id] # set it to the desired grouping

	return(df)

def assert_key_columns_present(df):
	"""Ensure that the DataFrame has essential columns and handle missing values.
    
    This function removes all special characters from the DataFrame column names
    and checks if the essential columns `conversation_num`, `message`, and 
    `speaker_nickname` are present. If any of these columns are missing, a 
    KeyError is raised. It also fills missing values in the essential columns 
    with default values.

    :param df: The DataFrame to check and process.
    :type df: pandas.DataFrame
    :raises KeyError: If one of `conversation_num`, `message`, or `speaker_nickname` columns is missing.
    """

	# remove all special characters from df
	df.columns = df.columns.str.replace('[^A-Za-z0-9_]', '', regex=True)

	# Assert that key columns are present
	if {'conversation_num', 'message', 'speaker_nickname'}.issubset(df.columns):
		print("Confirmed that data has `conversation_num`, `message`, and `speaker_nickname` columns!")
		# ensure no NA's in essential columns
		df['message'] = df['message'].fillna('')
		df['conversation_num'] = df['conversation_num'].fillna(0)
		df['speaker_nickname'] = df['speaker_nickname'].fillna(0)
	else:
		print("One of `conversation_num`, `message`, or `speaker_nickname` is missing! Raising error...")
		print("Columns available: ")
		print(df.columns)
		raise KeyError

def preprocess_text_lowercase_but_retain_punctuation(text):
	"""Convert the input text to lowercase while retaining punctuation.

    This function takes a string and converts all characters to lowercase,
    keeping any punctuation marks intact.

    :param text: The input text to process.
    :type text: str
    :return: The processed text with all characters in lowercase.
    :rtype: str
    """
	return(text.lower())

def preprocess_text(text):
	"""Preprocess text by removing non-alphanumeric characters and converting to lowercase.

    This function takes a string, removes any characters that are not letters, numbers, or spaces,
    and converts the remaining text to lowercase.

    :param text: The input text to process.
    :type text: str
    :return: The processed text containing only alphanumeric characters and spaces in lowercase.
    :rtype: str
    """
	return(re.sub(r"[^a-zA-Z0-9 ]+", '',text).lower())

def preprocess_naive_turns(chat_data):
	"""Combine adjacent rows of the same speaker in the same conversation and compress messages into a "turn".

    This function first generates a 'turn_id' for each chat message within the same conversation,
    indicating turns taken by the active speaker. It then combines messages with the same 'turn_id' 
    within each conversation to compress repeated messages from the same speaker.

    :param chat_data: The chat data to process.
    :type chat_data: pandas.DataFrame
    :return: The processed chat data with combined message turns.
    :rtype: pandas.DataFrame
    """
	turn_id_per_conv = chat_data.groupby(['conversation_num'], sort=False).apply(lambda df : get_turn_id(df))
	turn_id_per_conv = turn_id_per_conv.to_frame().reset_index().rename(columns={0:'turn_id'})
	chat_data = pd.concat([chat_data, turn_id_per_conv["turn_id"]], axis=1)
	
	# Use turn_id to compress messages with the same turn id per conversation
	chat_data = chat_data.groupby('conversation_num', sort=False).apply(lambda df : df.groupby('turn_id', as_index=False).apply(compress)).reset_index(drop=True)
	
	return chat_data

def get_turn_id(df):
	"""Generate turn IDs for a conversation to identify turns taken by speakers.

    This function compares the current speaker with the previous one to identify when a change in speaker occurs, 
    and then assigns a unique 'turn_id' that increments whenever the speaker changes within the conversation.

    :param df: The DataFrame containing chat data for a single conversation.
    :type df: pandas.DataFrame
    :return: A Series containing the turn IDs.
    :rtype: pandas.Series
    """
	df["speaker_nickname_x"] = df["speaker_nickname"].shift()
	return (df["speaker_nickname"] != df["speaker_nickname_x"]).cumsum()
	
def compress(turn_df):
	"""Combine messages in the same turn into a single message.

    This function takes a DataFrame representing messages in a single turn and
    concatenates their 'message' and 'message_lower_with_punc' columns into
    single strings if there are multiple messages in the same turn.

    :param turn_df: The DataFrame containing messages in a single turn.
    :type turn_df: pandas.DataFrame
    :return: A Series with combined messages for the turn.
    :rtype: pandas.Series
    """
	result = turn_df.iloc[0]
	if (len(turn_df) > 1):
		result['message'] = turn_df['message'].str.cat(sep=' ')
		result['message_lower_with_punc'] = turn_df['message_lower_with_punc'].str.cat(sep=' ')
	return result

def create_cumulative_rows(input_df, conversation_id, within_task = False):
	"""Generate cumulative rows for chat data to analyze conversations in context.

    This function takes chat-level data and duplicates rows to facilitate the analysis of conversations
    in the context of preceding chats. It enables the inclusion of chats from previous stages or tasks within
    the same conversation.

    NOTE: This function was created in the context of a multi-stage Empirica game (see: https://github.com/Watts-Lab/multi-task-empirica).
    It may not be generalizable to other conversational data, and will likely be deprecated in future versions.

    :param input_df: The DataFrame containing chat data.
    :type input_df: pandas.DataFrame
    :param conversation_id: The ID (e.g., stage or round) used for grouping the data.
    :type conversation_id: str
    :param within_task: Flag to determine whether to restrict the analysis to the same task (`roundId`), defaults to False.
    :type within_task: bool, optional
    :return: The processed DataFrame with cumulative rows added.
    :rtype: pandas.DataFrame
    """

	# If the conversation_id is the gameId, return as is -- no changes requred
	if(conversation_id == "gameId"): return input_df

	result_df = pd.DataFrame(columns=input_df.columns)

	# prev stageId
	prev_stageId = None

	# Iterate through rows
	for index, current_row in input_df.iterrows():
			
		# current stageId
		if current_row["stageId"] != prev_stageId: # we have transitioned to a new stageId

			prev_stageId = current_row["stageId"]

			if(conversation_id == 'stageId'):
				# Duplicate rows from all previous stageId's with the same 'gameId'
				if(within_task): # ensure roundId's are the same
					previous_rows = input_df.loc[(input_df['stageId'] != current_row['stageId']) & (input_df['timestamp'] < current_row['timestamp']) & (input_df['gameId'] == current_row['gameId']) & (input_df['roundId'] == current_row['roundId'])].copy()
				else:
					previous_rows = input_df.loc[(input_df['stageId'] != current_row['stageId']) & (input_df['timestamp'] < current_row['timestamp']) & (input_df['gameId'] == current_row['gameId'])].copy()
				if(not previous_rows.empty):
					previous_rows['cumulative_Id'] = current_row["stageId"]
					result_df = pd.concat([result_df, previous_rows], ignore_index=True)
			if(conversation_id == 'roundId'):
				# Duplicate rows from all previous roundId's with the same gameId
				previous_rows = input_df.loc[(input_df['roundId'] != current_row['roundId']) & (input_df['timestamp'] < current_row['timestamp']) & (input_df['gameId'] == current_row['gameId'])].copy()
				if(not previous_rows.empty):
					previous_rows['cumulative_Id'] = current_row["roundId"]
					result_df = pd.concat([result_df, previous_rows], ignore_index=True)

			cur_Id_rows = input_df.loc[(input_df[conversation_id] == current_row[conversation_id])].copy()
			cur_Id_rows['cumulative_Id'] = current_row[conversation_id]
			# Concatenate the current row to the result DataFrame
			result_df = pd.concat([result_df, cur_Id_rows], ignore_index=True).drop_duplicates()

	return result_df
