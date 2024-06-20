import numpy as np

# Defines simple text features on a text or conversation.

def count_words(text):
	""" Returns the number of words in a message.

	Args:
		text (str): The message (utterance) for which we are counting words.

	Returns:
		int: number of words
	"""
	return len(text.split())

def count_characters(text):
	""" Counts the number of characters in a message.

	Args:
		text (str): The message (utterance) for which we are counting characters.

	Retunrs:
		int: number of characters
	"""
	return len(str(text))

def count_messages(text):
	"""
	This function is trivial; by definition, each message counts as 1. However, at the conversation level, we use this function to count the total number of messages/utterance via aggregation.
	
	Args:
		text (str): The message (utterance); not used in the function.

	Return:
		int: 1 (as each message trivially contains 1 message)
	"""
	return(1)