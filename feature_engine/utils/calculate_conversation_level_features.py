"""
file: calculate_conversation_level_features.py
---
This file defines the ConvLevelFeaturesCalculator class using the modules defined in "features".
The intention behind this class is to use these modules and define any and all conv level features here. 
"""

# Importing modules from features
from features.gini_coefficient import *
from features.basic_features import *
from utils.summarize_chat_level_features import *
from features.get_all_DD_features import *


class ConversationLevelFeaturesCalculator:
    def __init__(self, chat_data: pd.DataFrame, conv_data: pd.DataFrame, vect_data: pd.DataFrame, input_columns:list) -> None:
        """
            This function is used to initialize variables and objects that can be used by all functions of this class.

		PARAMETERS:
			@param chat_data (pd.DataFrame): This is a pandas dataframe of the chat level features read in from the input dataset.
            @param conv_data (pd.DataFrame): This is a pandas dataframe of the conversation level features derived from the 
                                             chat level dataframe.
            @param input_columns (list): This is a list containing all the columns in the chat level features dataframe that 
                                         should not be summarized.
        """
        # Initializing variables
        self.chat_data = chat_data
        self.conv_data = conv_data
        self.vect_data = vect_data
        # Denotes the columns that can be summarized from the chat level, onto the conversation level.
        self.input_columns = list(input_columns)
        self.input_columns.append('conversation_num')
        self.columns_to_summarize = [column for column in self.chat_data.columns \
                                     if (column not in self.input_columns) and pd.api.types.is_numeric_dtype(self.chat_data[column])]

    def calculate_conversation_level_features(self) -> pd.DataFrame:
        """
			This is the main driver function for this class.

		RETURNS:
			(pd.DataFrame): The conversation level dataset given to this class during initialization along with 
							new columns for each conv level feature.
        """
        # Get gini based features
        self.get_gini_features()
        # Get summary statistics by aggregating chat level features
        self.get_conversation_level_summary_statistics_features()
        # Get 4 discursive features (discursive diversity, variance in DD, incongruent modulation, within-person discursive range)
        self.get_discursive_diversity_features()

        return self.conv_data

    def get_gini_features(self) -> None:
        """
            This function is used to calculate the gini index for each conversation 
            based on the word level and character level information.
        """
        # Gini for #Words
        self.conv_data = pd.merge(
            left=self.conv_data,
            right=get_gini(self.chat_data, "num_words"),
            on=['conversation_num'],
            how="inner"
        )

        # Gini for #Characters
        self.conv_data = pd.merge(
            left=self.conv_data,
            right=get_gini(self.chat_data, "num_chars"),
            on=['conversation_num'],
            how="inner"
        )

    def get_conversation_level_summary_statistics_features(self) -> None:
        """
            This function is used to aggregate the summary statistics from 
            chat level features to conversation level features.
            Specifically, it looks at the mean and standard deviations at message and word level.
        """
        # For each summarizable feature
        for column in self.columns_to_summarize:
            # Average/Mean of feature across the Conversation
            self.conv_data = pd.merge(
                left=self.conv_data,
                right=get_average(self.chat_data, column, 'average_'+column),
                on=['conversation_num'],
                how="inner"
            )

            # Standard Deviation of feature across the Conversation
            self.conv_data = pd.merge(
                left=self.conv_data,
                right=get_stdev(self.chat_data, column, 'stdev_'+column),
                on=['conversation_num'],
                how="inner"
            )

            # Minima for the feature across the Conversation
            self.conv_data = pd.merge(
                left=self.conv_data,
                right=get_min(self.chat_data, column, 'min_'+column),
                on=['conversation_num'],
                how="inner"
            )

            # Maxima for the feature across the Conversation
            self.conv_data = pd.merge(
                left=self.conv_data,
                right=get_max(self.chat_data, column, 'max_'+column),
                on=['conversation_num'],
                how="inner"
            )
    
    def get_discursive_diversity_features(self) -> None:
        """
            This function is used to calculate the discursive diversity for each conversation 
            based on the word embeddings (SBERT) and chat level information.
        """
        self.conv_data = pd.merge(
            left=self.conv_data,
            right=get_DD_features(self.chat_data, self.vect_data),
            on=['conversation_num'],
            how="inner"
        )
