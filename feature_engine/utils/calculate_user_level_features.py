# Importing modules from features
from utils.summarize_features import get_user_sum_dataframe, get_user_average_dataframe
from features.get_user_network import *
from features.user_centroids import *


class UserLevelFeaturesCalculator:
    """
    Initialize variables and objects used by the UserLevelFeaturesCalculator class.

    This class uses various feature modules to define user- (speaker) level features. It reads input data and
    initializes variables required to compute the features.

    :param chat_data: Pandas dataframe of chat-level features read from the input dataset
    :type chat_data: pd.DataFrame
    :param user_data: Pandas dataframe of user-level features derived from the chat-level dataframe
    :type user_data: pd.DataFrame
    :param vect_data: Pandas dataframe of message embeddings corresponding to each instance of the chat data
    :type vect_data: pd.DataFrame
    :param input_columns: List of columns in the chat-level features dataframe that should not be summarized
    :type input_columns: list
    """
    def __init__(self, chat_data: pd.DataFrame, user_data: pd.DataFrame, vect_data: pd.DataFrame, input_columns:list) -> None:

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
        Main driver function for creating user-level features.

        This function computes various user-level features by aggregating chat-level data,
        and appends them as new columns to the input user-level data.

        :return: The user-level dataset with new columns for each user-level feature
        :rtype: pd.DataFrame
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
        Aggregate summary statistics from chat-level features that need to be summed together.

        Features for which summing makes sense include:
        - Word count (total number of words)
        - Character count
        - Message count
        - Function word accommodation

        This function calculates and merges the summed features into the user-level data.

        :return: None
        :rtype: None
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
        Aggregate summary statistics by calculating average user-level features from chat-level features.

        This function calculates and merges the average features into the user-level data.

        :return: None
        :rtype: None
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
        Calculate the centroid of each user's chats in a given conversation for future discursive metric calculations.

        This function computes and appends the mean embedding (centroid) of each user's chats to the user-level data.

        :return: None
        :rtype: None
        """
        self.user_data['mean_embedding'] = get_user_centroids(self.chat_data, self.vect_data)

    def get_user_network(self) -> None:
        """
        Get the user list per user per conversation.

        This function calculates and appends the list of other users in a given conversation to the user-level data.

        :return: None
        :rtype: None
        """
        self.user_data = pd.merge(
                left=self.user_data,
                right=get_user_network(self.user_data),
                on=['conversation_num', 'speaker_nickname'],
                how="inner"
            )