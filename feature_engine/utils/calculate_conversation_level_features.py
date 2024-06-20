# Importing modules from features
from utils.gini_coefficient import *
from features.basic_features import *
from utils.summarize_features import *
from utils.preprocess import *
from features.get_all_DD_features import *
from features.turn_taking_features import*
from features.burstiness import *
from features.information_diversity import *


class ConversationLevelFeaturesCalculator:
    """
    Initialize variables and objects used by the ConversationLevelFeaturesCalculator class.

    This class uses various feature modules to define conversation-level features. It reads input data and
    initializes variables required to compute the features.

    :param chat_data: Pandas dataframe of chat-level features read from the input dataset
    :type chat_data: pd.DataFrame
    :param user_data: Pandas dataframe of user-level features derived from the chat-level dataframe
    :type user_data: pd.DataFrame
    :param conv_data: Pandas dataframe of conversation-level features derived from the chat-level dataframe
    :type conv_data: pd.DataFrame
    :param vect_data: Pandas dataframe of processed vectors derived from the chat-level dataframe
    :type vect_data: pd.DataFrame
    :param vector_directory: Directory where vector files are stored
    :type vector_directory: str
    :param input_columns: List of columns in the chat-level features dataframe that should not be summarized
    :type input_columns: list
        """
    def __init__(self, chat_data: pd.DataFrame, 
                        user_data: pd.DataFrame, 
                        conv_data: pd.DataFrame, 
                        vect_data: pd.DataFrame, 
                        vector_directory: str, 
                        input_columns:list) -> None:
    
        # Initializing variables
        self.chat_data = chat_data
        self.user_data = user_data
        self.conv_data = conv_data
        self.vect_data = vect_data
        self.vector_directory = vector_directory
        # Denotes the columns that can be summarized from the chat level, onto the conversation level.
        self.input_columns = list(input_columns)
        self.input_columns.append('conversation_num')
        self.columns_to_summarize = [column for column in self.chat_data.columns \
                                     if (column not in self.input_columns) and pd.api.types.is_numeric_dtype(self.chat_data[column])]
        self.summable_columns = ["num_words", "num_chars", "num_messages"]
        
    def calculate_conversation_level_features(self) -> pd.DataFrame:
        """
        Main driver function for creating conversation-level features.

        This function computes various conversation-level features by aggregating chat-level and user-level features,
        and appends them as new columns to the input conversation-level data.

        :return: The conversation-level dataset with new columns for each conversation-level feature
        :rtype: pd.DataFrame
        """

        # Get turn taking index by aggregating chat level totals, pass in CHAT LEVEL FEATURES
        self.get_turn_taking_features()
        print("Generated turn taking index.")

        # Get gini based features by aggregating user-level totals, pass in USER LEVEL FEATURES
        self.get_gini_features()
        print("Generated gini features.")

        # Get summary statistics by aggregating chat level features, pass in CHAT LEVEL FEATURES
        self.get_conversation_level_aggregates()
        print("Generated chat aggregates.")

        # Get summary statistics by aggregating user level features, pass in USER LEVEL FEATURES 
        self.get_user_level_aggregates()
        print("Generated user aggregates.")
        
        # Get 4 discursive features (discursive diversity, variance in DD, incongruent modulation, within-person discursive range)
        self.get_discursive_diversity_features()

        # Get team burstiness coefficient using chat level temporal features
        self.calculate_team_burstiness()

        # Get team's information diversity score
        self.calculate_info_diversity()

        return self.conv_data

    def get_turn_taking_features(self) -> None:
        """
        Calculate the turn-taking index in the conversation.

        This function merges turn-taking features into the conversation-level data.

        :return: None
        :rtype: None
        """

        self.conv_data = pd.merge(
            left=self.conv_data,
            right=get_turn(self.chat_data.copy()),
            on=["conversation_num"],
            how="inner"
        )

    def get_gini_features(self) -> None:
        """
        Calculate the Gini index for relevant features in the conversation.

        This function computes the Gini index for features involving counts, such as:
        - Word count
        - Character count
        - Message count
        - Function word accommodation

        The Gini index is then merged into the conversation-level data.

        :return: None
        :rtype: None
        """
        for column in self.summable_columns:
            
            self.conv_data = pd.merge(
                left=self.conv_data,
                right=get_gini(self.user_data.copy(), "sum_"+column), # this applies to the summed columns in user_data, which matches the above
                on=['conversation_num'],
                how="inner"
            )

    def get_conversation_level_aggregates(self) -> None:
        """
        Aggregate summary statistics from chat-level features to conversation-level features.

        This function calculates and merges the following aggregation functions for each summarizable feature:
        - Average/Mean
        - Standard Deviation
        - Minimum
        - Maximum

        For countable features (e.g., num_words, num_chars, num_messages), it also calculates and merges the sum.

        :return: None
        :rtype: None
        """

        # For each summarizable feature
        for column in self.columns_to_summarize:
            
            # Average/Mean of feature across the Conversation
            self.conv_data = pd.merge(
                left=self.conv_data,
                right=get_average(self.chat_data.copy(), column, 'average_'+column),
                on=['conversation_num'],
                how="inner"
            )

            # Standard Deviation of feature across the Conversation
            self.conv_data = pd.merge(
                left=self.conv_data,
                right=get_stdev(self.chat_data.copy(), column, 'stdev_'+column),
                on=['conversation_num'],
                how="inner"
            )

            # Minima for the feature across the Conversation
            self.conv_data = pd.merge(
                left=self.conv_data,
                right=get_min(self.chat_data.copy(), column, 'min_'+column),
                on=['conversation_num'],
                how="inner"
            )

            # Maxima for the feature across the Conversation
            self.conv_data = pd.merge(
                left=self.conv_data,
                right=get_max(self.chat_data.copy(), column, 'max_'+column),
                on=['conversation_num'],
                how="inner"
            )

        # Do this only for the columns that make sense (e.g., countable things)
        for column in self.summable_columns:
            # Sum for the feature across the Conversation
            self.conv_data = pd.merge(
                left=self.conv_data,
                right=get_sum(self.chat_data.copy(), column, 'sum_'+column),
                on=['conversation_num'],
                how="inner"
            )
    
    def get_user_level_aggregates(self) -> None:
        """
        Aggregate summary statistics from user-level features to conversation-level features.

        This function calculates and merges the following aggregation functions for each user-level feature:
        - Average/Mean of summed user-level features
        - Standard Deviation of summed user-level features
        - Minimum of summed user-level features
        - Maximum of summed user-level features
        - Average/Mean of averaged user-level features
        - Standard Deviation of averaged user-level features
        - Minimum of averaged user-level features
        - Maximum of averaged user-level features

        :return: None
        :rtype: None
        """

        # Sum Columns were created using self.get_user_level_summed_features()
        for column in self.columns_to_summarize:
            
            # Average/Mean of User-Level Feature
            self.conv_data = pd.merge(
                left=self.conv_data,
                right=get_average(self.user_data.copy(), "sum_"+column, 'average_user_sum_'+column),
                on=['conversation_num'],
                how="inner"
            )

            # Standard Deviation of User-Level Feature
            self.conv_data = pd.merge(
                left=self.conv_data,
                right=get_stdev(self.user_data.copy(), "sum_"+column, 'stdev_user_sum_'+column),
                on=['conversation_num'],
                how="inner"
            )

            # Minima of User-Level Feature
            self.conv_data = pd.merge(
                left=self.conv_data,
                right=get_min(self.user_data.copy(), "sum_"+column, 'min_user_sum_'+column),
                on=['conversation_num'],
                how="inner"
            )

            # Maxima of User-Level Feature
            self.conv_data = pd.merge(
                left=self.conv_data,
                right=get_max(self.user_data.copy(), "sum_"+column, 'max_user_sum_'+column),
                on=['conversation_num'],
                how="inner"
            )

        # Average Columns were created using self.get_user_level_averaged_features()
        for column in self.columns_to_summarize:
            
            # Average/Mean of User-Level Feature
            self.conv_data = pd.merge(
                left=self.conv_data,
                right=get_average(self.user_data.copy(), "average_"+column, 'average_user_avg_'+column),
                on=['conversation_num'],
                how="inner"
            )

            # Standard Deviation of User-Level Feature
            self.conv_data = pd.merge(
                left=self.conv_data,
                right=get_stdev(self.user_data.copy(), "average_"+column, 'stdev_user_avg_'+column),
                on=['conversation_num'],
                how="inner"
            )

            # Minima of User-Level Feature
            self.conv_data = pd.merge(
                left=self.conv_data,
                right=get_min(self.user_data.copy(), "average_"+column, 'min_user_avg_'+column),
                on=['conversation_num'],
                how="inner"
            )

            # Maxima of User-Level Feature
            self.conv_data = pd.merge(
                left=self.conv_data,
                right=get_max(self.user_data.copy(), "average_"+column, 'max_user_avg_'+column),
                on=['conversation_num'],
                how="inner"
            )

    
    def get_discursive_diversity_features(self) -> None:
        """
        Calculate discursive diversity features for each conversation.

        This function computes discursive diversity based on the word embeddings (SBERT) 
        and chat-level information, and merges the features into the conversation-level data.

        :return: None
        :rtype: None
        """
        self.conv_data = pd.merge(
            left=self.conv_data,
            right=get_DD_features(self.chat_data, self.vect_data),
            on=['conversation_num'],
            how="inner"
        )
    
      
    def calculate_team_burstiness(self) -> None:
        """
        Calculate the team burstiness coefficient.

        This function computes the team burstiness coefficient by looking at the differences 
        in standard deviation and mean of the time intervals between chats, and merges the 
        results into the conversation-level data.

        :return: None
        :rtype: None
        """
        if {'time_diff'}.issubset(self.chat_data.columns):
            self.conv_data = pd.merge(
            left = self.conv_data,
            right = get_team_burstiness(self.chat_data, "time_diff"),
            on = ['conversation_num'],
            how = "inner"
        )
    
    def calculate_info_diversity(self) -> None:
        """
        Calculate an information diversity score for the team.

        This function computes the information diversity score by looking at the cosine 
        similarity between the mean topic vector of the team and each message's topic vectors, 
        and merges the results into the conversation-level data.

        :return: None
        :rtype: None
        """
        self.conv_data = pd.merge(
            left = self.conv_data,
            right = get_info_diversity(self.chat_data),
            on = ['conversation_num'],
            how = "inner"
        )