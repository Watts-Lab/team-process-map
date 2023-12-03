import re
import pandas as pd

def preprocess_conversation_columns(df, conversation_id = None, cumulative_grouping = False, within_task = False):
	# remove all special characters from df
	df.columns = df.columns.str.replace('[^A-Za-z0-9_]', '', regex=True)
	
	# If data is grouped by batch/round, add a conversation num
	if {'batch_num', 'round_num'}.issubset(df.columns):
		df['conversation_num'] = df.groupby(['batch_num', 'round_num']).ngroup()
		df = df[df.columns.tolist()[-1:] + df.columns.tolist()[0:-1]] # make the new column first
	if ({'gameId', 'roundId', 'stageId'}.issubset(df.columns) and conversation_id is not None):
		
		if(cumulative_grouping):

			try:
				assert(conversation_id == "stageId")
			except AssertionError:
				print("When analyzing data cumulatively, the `conversation_id` must be at the `stageId` level.")

			df = create_cumulative_rows(input_df, within_task)
			df['conversation_num'] = df['cumulative_stageId'] # set it to be the cumulative grouping
		else:
			df['conversation_num'] = df[conversation_id]

	return(df)

def assert_key_columns_present(df):
	# remove all special characters from df
	df.columns = df.columns.str.replace('[^A-Za-z0-9_]', '', regex=True)

	# Assert that key columns are present
	if {'conversation_num', 'message', 'speaker_nickname'}.issubset(df.columns):
		print("Confirmed that data has `conversation_num`, `message`, and `speaker_nickname` columns!")
	else:
		print("One of `conversation_num`, `message`, or `speaker_nickname` is missing! Raising error...")
		print("Columns available: ")
		print(df.columns)
		raise KeyError

def preprocess_text_lowercase_but_retain_punctuation(text):
  	# Only turns the text lowercase
	return(text.lower())

def preprocess_text(text):
  	# For each individual message: preprocess to remove anything that is not an alphabet or number from the string
	return(re.sub(r"[^a-zA-Z0-9 ]+", '',text).lower())

def preprocess_naive_turns(chat_data):
	# Combine adjacent rows of the same speaker in the same conversation
	
	# Generate 'turn_id' per chat per conversation, dependent on the active speaker
	turn_id_per_conv = chat_data.groupby(['conversation_num'], sort=False).apply(lambda df : get_turn_id(df))
	turn_id_per_conv = turn_id_per_conv.to_frame().reset_index().rename(columns={0:'turn_id'})
	chat_data = pd.concat([chat_data, turn_id_per_conv["turn_id"]], axis=1)
	
	# Use turn_id to compress messages with the same turn id per conversation
	chat_data = chat_data.groupby('conversation_num', sort=False).apply(lambda df : df.groupby('turn_id', as_index=False).apply(compress)).reset_index(drop=True)
	
	return chat_data

def get_turn_id(df) :
    df["speaker_nickname_x"] = df["speaker_nickname"].shift()
    return (df["speaker_nickname"] != df["speaker_nickname_x"]).cumsum()
    
def compress(turn_df):
    result = turn_df.iloc[0]
    if (len(turn_df) > 1):
        result['message'] = turn_df['message'].str.cat(sep=' ')
        result['message_lower_with_punc'] = turn_df['message_lower_with_punc'].str.cat(sep=' ')
    return result

def create_cumulative_rows(input_df, within_task = False):
    """
    function: create_cumulative_rows

    This function takes a chat-level dataframe and duplicates rows such that we can analyze
    convesations in the context of what came before.

    For example, rather than analyzing only the chats from a single stage, this function makes it possible
    to also incorporate chats from previous stages / tasks in the same conversation.

    @param within_task (defaults to False). This parameter determines whether we want to restrict
    the analysis only to chats that were of the same task (e.g., same `roundId`). By default, we look at 
    every chat that came before, regardless of the task.
    """
    result_df = pd.DataFrame(columns=input_df.columns)

    # prev stageID
    prev_stageID = None

    # Iterate through rows
    for index, current_row in input_df.iterrows():
            
        # current stageID
        if current_row["stageID"] != prev_stageID: # we have transitioned to a new stageID

            prev_stageID = current_row["stageID"]

            # Duplicate rows from all previous 'stageID's with the same 'gameId'
            # Note: add `& (input_df['roundId'] == current_row['roundId'])` here to restrict this to the same round
            if(within_task):
                previous_rows = input_df.loc[(input_df['stageID'] != current_row['stageID']) & (input_df['timestamp'] < current_row['timestamp']) & (input_df['gameId'] == current_row['gameId']) & (input_df['roundId'] == current_row['roundId'])].copy()
            else:
                previous_rows = input_df.loc[(input_df['stageID'] != current_row['stageID']) & (input_df['timestamp'] < current_row['timestamp']) & (input_df['gameId'] == current_row['gameId'])].copy()
            if(not previous_rows.empty):
                previous_rows['cumulative_stageId'] = current_row["stageID"]
                result_df = pd.concat([result_df, previous_rows], ignore_index=True)

            cur_stageID_rows = input_df.loc[(input_df['stageID'] == current_row['stageID'])].copy()
            cur_stageID_rows['cumulative_stageId'] = current_row["stageID"]
            # Concatenate the current row to the result DataFrame
            result_df = pd.concat([result_df, cur_stageID_rows], ignore_index=True)

    return result_df
