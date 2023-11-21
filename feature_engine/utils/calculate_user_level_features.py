"""
file: calculate_user_level_features.py
---
This file defines the UserLevelFeaturesCalculator class using the modules defined in "features".

The intention behind this class is to use these modules and define any and all user level features here. 
"""

# Importing modules from features
from utils.summarize_features import get_user_sum_dataframe, get_user_average_dataframe
from features.get_user_network import *
from features.user_centroids import *


class UserLevelFeaturesCalculator:
    def __init__(self, chat_data: pd.DataFrame, user_data: pd.DataFrame, vect_data: pd.DataFrame, input_columns:list) -> None:
        """
            This function is used to initialize variables and objects that can be used by all functions of this class.

		PARAMETERS:
			@param chat_data (pd.DataFrame): This is a pandas dataframe of the chat level features read in from the input dataset.
            @param user_data (pd.DataFrame): This is a pandas dataframe of the user level features derived from the chat level dataframe.
            @param vect_data (pd.DataFrame): This is a pandas dataframe of the message embeddings correlating with each instance of the chat csv. 
            @param input_columns (list): This is a list containing all the columns in the chat level features dataframe that 
                                         should not be summarized.
        """
        # Initializing variables
        self.chat_data = chat_data
        self.user_data = user_data
        self.vect_data = vect_data

        # Denotes the columns that can be summarized from the chat level, onto the conversation level.
        self.input_columns = list(input_columns)
        self.input_columns.append('conversation_num')
        self.columns_to_summarize = [column for column in self.chat_data.columns \
                                     if (column not in self.input_columns) and pd.api.types.is_numeric_dtype(self.chat_data[column])]

    def calculate_user_level_features(self) -> pd.DataFrame:
        """
			This is the main driver function for this class.

		RETURNS:
			(pd.DataFrame): The conversation level dataset given to this class during initialization along with 
							new columns for each conv level feature.
        """

        # Get average features for all features
        self.get_user_level_averaged_features()
        
        # Get total counts for all features
        self.get_user_level_summed_features()
        
        # Get 4 discursive features (discursive diversity, variance in DD, incongruent modulation, within-person discursive range)
        # self.get_centroids()

        # Get list of other users in a given conversation
        self.get_user_network()

        return self.user_data


    def get_user_level_summary_statistics_features(self) -> None:
        """
            This function is used to aggregate the summary statistics from 
            chat level features to user level features.
            
            There are many possible ways to aggregate user level features, e.g.:
            - Mean of all chats by a given user;
            - Max of all chats by a given user;
            - Weighted mean (e.g., looking at different time points?)
            ... and so on.

            This is an open question, so we are putting a TODO here.
        """
        pass

    def get_user_level_summed_features(self) -> None:
        """
            This function is used to aggregate the summary statistics from 
            chat level features that need to be SUMMED together. Featuers for which this makes sense are:

            - word count (e.g., total number of words)
            - character count
            - message count
            - function_word_accommodation

            (In essence, these all represent _counts_ of something, for which it makes sense to get a "total")
        """
        # For each summarizable feature
        for column in self.columns_to_summarize:
            # Sum of feature across the Conversation
            self.user_data = pd.merge(
                left=self.user_data,
                right=get_user_sum_dataframe(self.chat_data, column),
                on=['conversation_num', 'speaker_nickname'],
                how="inner"
            )

    def get_user_level_averaged_features(self) -> None:
        """
            This function is used to aggregate the summary statistics from 
            chat level features.
        """
        # For each summarizable feature
        for column in self.columns_to_summarize:
            # Average/Mean of feature across the Conversation
            self.user_data = pd.merge(
                left=self.user_data,
                right=get_user_average_dataframe(self.chat_data, column),
                on=['conversation_num', 'speaker_nickname'],
                how="inner"
            )

    def get_centroids(self) -> None:
        """
        This function is used to get the centroid of each user's chats in a given conversation to be used for future discursive metric calculations. 
        
        """
        self.user_data['mean_embedding'] = get_user_centroids(self.chat_data, self.vect_data)

    def get_user_network(self) -> None:
        '''
        This function gets the user_list per user per conversation.

        '''
        self.user_data = pd.merge(
                left=self.user_data,
                right=get_user_network(self.user_data),
                on=['conversation_num', 'speaker_nickname'],
                how="inner"
            )


# FEATURES FOR USER LEVEL

# 1. Semantic modulation
# MAXIMUM shift in speech style within a transition between chunks
# AVERAGE shift in speech style across chunks
# VARIANCE shift across all chunks (high variance --> inconsistent shifts vs consistent shifts)

# 2. Summarize chat level features for speakers (get count dataframe)
# words, characters, LIWC, ConvoKit, BERT
# GETTING CENTROIDS FOR CONVERSATION --> discursive diversity (per conversation)
# --> may have to store this information in separate file for readability?

# 3. Inputs for a model
# Currently, all the inputs are in the conversational-level output. All chat level features are aggregated into min, average, max, and stdev across all speakers in a given conversation. For example, MAX positivity represents the speaker with the highest positivity score across all chats in its conversation.

 
