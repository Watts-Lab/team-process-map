import re
import pandas as pd

def preprocess_conversation_columns(df):
	# remove all special characters from df
	df.columns = df.columns.str.replace('[^A-Za-z0-9_]', '', regex=True)
	
	# If data is grouped by batch/round, add a conversation num
	if {'batch_num', 'round_num'}.issubset(df.columns):
		df['conversation_num'] = df.groupby(['batch_num', 'round_num']).ngroup()
		df = df[df.columns.tolist()[-1:] + df.columns.tolist()[0:-1]] # make the new column first

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



