"""
file: feature_builder.py
---
This file defines the FeatureBuilder class using the modules defined in "utils" and "features".
The intention behind this class is to use these modules and:
- Preprocess the incoming dataset defined in an input file path.
- Create chat level features -> Use the moduled in "utils" and "features" to create features 
                                on each chat message in the dataset (like word count, character count etc.).
- Create conversation level features -> These can come from 2 sources:
                                        - By aggregating the chat level features
                                        - By defining new features specifically applicable for conversations
- Save the chat and conversation level features in the output path specified.
"""

# 3rd Party Imports
import pandas as pd
import re
import numpy as np
from pathlib import Path

# Imports from feature files and classes
# from utils.summarize_chat_level_features import *
from utils.calculate_chat_level_features import ChatLevelFeaturesCalculator
from utils.calculate_user_level_features import UserLevelFeaturesCalculator
from utils.calculate_conversation_level_features import ConversationLevelFeaturesCalculator
from utils.preprocess import *
from utils.check_embeddings import *

class FeatureBuilder:
    def __init__(
            self, 
            input_file_path: str, 
            output_file_path_chat_level: str, 
            output_file_path_user_level: str,
            output_file_path_conv_level: str,
            analyze_first_pct: list = [1.0], 
            turns: bool=True
        ) -> None:
        """
            This function is used to define variables used throughout the class.

        PARAMETERS:
            @param input_file_path (str): File path of the input csv dataset (assumes that the '.csv' suffix is added)
            @param output_file_path_chat_level (str): Path where the output csv file is to be generated 
                                                      (assumes that the '.csv' suffix is added)
            @param output_file_path_conv_level (str): Path where the output csv file is to be generated 
                                                      (assumes that the '.csv' suffix is added)
            @param analyze_first_pct (list of floats): Analyze the first X% of the data.
                This parameter is useful because the earlier stages of the conversation may be more predictive than
                the later stages. Thus, researchers may wish to analyze only the first X% of the conversation data
                and compare the performance with using the full dataset.
                This defaults to a single list containing the full dataset.
        """
        #  Defining input and output paths.
        self.input_file_path = input_file_path
        print("Initializing Featurization for " + self.input_file_path + " ...")
        self.output_file_path_conv_level = output_file_path_conv_level
        self.output_file_path_user_level = output_file_path_user_level

        # Set first pct of conversation you want to analyze
        assert(all(0 <= x <= 1 for x in analyze_first_pct)) # first, type check that this is a list of numbers between 0 and 1
        self.first_pct = analyze_first_pct

        # Reading chat level data (this is available in the input file path directly).
        self.chat_data = pd.read_csv(self.input_file_path, encoding='mac_roman')

        # Preprocess chat data
        self.turns = turns
        self.preprocess_chat_data(col="message", turns=self.turns)

        # Input columns are the columns that come in the raw chat data
        self.input_columns = self.chat_data.columns

        # Set all paths for vector retrieval (contingent on turns)
        df_type = "turns" if self.turns else "chats"
        self.vect_path = re.sub('raw_data', 'vectors/sentence/' + df_type, input_file_path)
        self.bert_path = re.sub('raw_data', 'vectors/sentiment/' + df_type, input_file_path)
        self.output_file_path_chat_level = re.sub('chat', 'turn', output_file_path_chat_level) if self.turns else output_file_path_chat_level

        # Check + generate embeddings
        check_embeddings(self.chat_data, self.vect_path, self.bert_path)

        self.vect_data = pd.read_csv(self.vect_path, encoding='mac_roman')
        self.bert_sentiment_data = pd.read_csv(self.bert_path, encoding='mac_roman').drop('Unnamed: 0', axis=1)

        # Deriving the base conversation level dataframe.
        # This is the number of unique conversations (and, in conversations with multiple levels, the number of
        # unique rows across "batch_num", and "round_num".)
        # Assume that "conversation_num" is the primary key for this table.
        self.conv_data = self.chat_data[['conversation_num']].drop_duplicates()

    def set_self_conv_data(self) -> None:
        """
        Deriving the base conversation level dataframe.
        Set Conversation Data around `conversation_num` once preprocessing completes.
        We need to select the first TWO columns, as column 1 is the 'index' and column 2 is 'conversation_num'
        """        
        self.conv_data = self.chat_data[['conversation_num']].drop_duplicates()

    def merge_conv_data_with_original(self) -> None:
        # Here, drop the message and speaker nickname (which do not matter at conversation level)
        orig_data = preprocess_conversation_columns(pd.read_csv(self.input_file_path, encoding='mac_roman')).drop(columns=['message', 'speaker_nickname'])
        orig_conv_data = orig_data.groupby(["conversation_num"]).nth(0).reset_index() # get 1st item (all conv items are the same)
        final_conv_output = pd.merge(
            left= self.conv_data,
            right = orig_conv_data,
            on=['conversation_num'],
            how="left"
        ).drop_duplicates()

        self.conv_data = final_conv_output

        # drop index column, if present
        if {'index'}.issubset(self.conv_data.columns):
            self.conv_data = self.conv_data.drop(columns=['index'])

    def featurize(self, col: str="message") -> None:
        """
            This is the main driver function of this class.
        
        PARAMETERS:
            @param col (str): (Default value: "message")
                              This is a parameter passed onto the preprocessing modules 
                              so as to identify the columns to preprocess.
        """
        # Step 1. Create chat level features.
        print("Generating Chat Level Features ...")
        self.chat_level_features()

        # Things to store before we loop through truncations
        self.chat_data_complete = self.chat_data # store complete chat data
        self.output_file_path_user_level_original = self.output_file_path_user_level
        self.output_file_path_chat_level_original = self.output_file_path_chat_level
        self.output_file_path_conv_level_original = self.output_file_path_conv_level

        # Step 2.
        # Run the chat-level features once, then produce different summaries based on 
        # user specification.
        for percentage in self.first_pct: 
            # Reset chat, conv, and user objects
            self.chat_data = self.chat_data_complete
            self.user_data = self.chat_data[['conversation_num', 'speaker_nickname']].drop_duplicates()
            self.set_self_conv_data()

            print("Generating features for the first " + str(percentage*100) + "% of messages...")
            self.get_first_pct_of_chat(percentage)
            
            # update output paths based on truncation percentage to save in a designated folder
            if percentage != 1: # special folders for when the percentage is partial
                self.output_file_path_user_level = re.sub('/output/', '/output/first_' + str(int(percentage*100)) + "/", self.output_file_path_user_level_original)
                self.output_file_path_chat_level = re.sub('/output/', '/output/first_' + str(int(percentage*100)) + "/", self.output_file_path_chat_level_original)
                self.output_file_path_conv_level = re.sub('/output/', '/output/first_' + str(int(percentage*100)) + "/", self.output_file_path_conv_level_original)
            else:
                self.output_file_path_user_level = self.output_file_path_user_level_original
                self.output_file_path_chat_level = self.output_file_path_chat_level_original
                self.output_file_path_conv_level = self.output_file_path_conv_level_original
            
            # Make it possible to create folders if they don't exist
            Path(self.output_file_path_user_level).parent.mkdir(parents=True, exist_ok=True)
            Path(self.output_file_path_chat_level).parent.mkdir(parents=True, exist_ok=True)
            Path(self.output_file_path_conv_level).parent.mkdir(parents=True, exist_ok=True)

            # Step 3a. Create user level features.
            print("Generating User Level Features ...")
            self.user_level_features()

            # Step 3b. Create conversation level features.
            print("Generating Conversation Level Features ...")
            self.conv_level_features()
            self.merge_conv_data_with_original()
            
            # Step 4. Write the feartures into the files defined in the output paths.
            print("All Done!")
            self.save_features()

    def preprocess_chat_data(self, col: str="message", turns=False) -> None:
        """
            This function is used to call all the preprocessing modules needed to clean the text.
        
        PARAMETERS:
            @param col (str): (Default value: "message")
                              This is used to identify the columns to preprocess.
        """
       
        # create the appropriate grouping variables and assert the columns are present
        self.chat_data = preprocess_conversation_columns(self.chat_data)
        assert_key_columns_present(self.chat_data)

        # create new column that retains punctuation
        self.chat_data["message_lower_with_punc"] = self.chat_data[col].astype(str).apply(preprocess_text_lowercase_but_retain_punctuation)
    
        # Preprocessing the text in `col` and then overwriting the column `col`.
        # TODO: We should probably use classes to abstract preprocessing module as well?
        self.chat_data[col] = self.chat_data[col].astype(str).apply(preprocess_text)

        if (turns):
            self.chat_data = preprocess_naive_turns(self.chat_data)

    def chat_level_features(self) -> None:
        """
            This function instantiates and uses the ChatLevelFeaturesCalculator to create the chat level features 
            and add them into the `self.chat_data` dataframe.
        """
        # Instantiating.
        chat_feature_builder = ChatLevelFeaturesCalculator(
            chat_data = self.chat_data,
            bert_sentiment_data = self.bert_sentiment_data
        )
        # Calling the driver inside this class to create the features.
        self.chat_data = chat_feature_builder.calculate_chat_level_features()
        # Remove special characters in column names
        self.chat_data.columns = ["".join(c for c in col if c.isalnum() or c == '_') for col in self.chat_data.columns]

    def get_first_pct_of_chat(self, percentage) -> None:
        """
            This function truncates each conversation to the first X% of rows.
        """
        chat_grouped = self.chat_data.groupby('conversation_num')
        num_rows_to_retain = pd.DataFrame(np.ceil(chat_grouped.size() * percentage)).reset_index()
        chat_truncated = pd.DataFrame()
        for conversation_num, num_rows in num_rows_to_retain.itertuples(index=False):
            chat_truncated = pd.concat([chat_truncated,chat_grouped.get_group(conversation_num).head(int(num_rows))], ignore_index = True)

        self.chat_data = chat_truncated

    def user_level_features(self) -> None:
        """
            This function instantiates and uses the UserLevelFeaturesCalculator to create the 
            user level features and add them into the `self.user_data` dataframe.
        """
        # Instantiating.
        self.user_data = preprocess_conversation_columns(self.user_data)
        user_feature_builder = UserLevelFeaturesCalculator(
            chat_data = self.chat_data, 
            user_data = self.user_data,
            vect_data= self.vect_data,
            input_columns = self.input_columns
        )
        
        # Calling the driver inside this class to create the features.
        self.user_data = user_feature_builder.calculate_user_level_features()
        # Remove special characters in column names
        self.user_data.columns = ["".join(c for c in col if c.isalnum() or c == '_') for col in self.user_data.columns]

    def conv_level_features(self) -> None:
        """
            This function instantiates and uses the ConversationLevelFeaturesCalculator to create the 
            conversation level features and add them into the `self.conv_data` dataframe.
        """
        # Instantiating.
        self.conv_data = preprocess_conversation_columns(self.conv_data)
        conv_feature_builder = ConversationLevelFeaturesCalculator(
            chat_data = self.chat_data, 
            user_data = self.user_data,
            conv_data = self.conv_data,
            vect_data = self.vect_data,
            input_columns = self.input_columns
        )
        # Calling the driver inside this class to create the features.
        self.conv_data = conv_feature_builder.calculate_conversation_level_features()

    def save_features(self) -> None:
        """
            This function simply saves the files in the respective output file paths provided during initialization.
        """
        # TODO: For now this function is very trivial. We will finilize the output formats (with date-time info etc) 
        # and control the output mechanism through this function.
        self.chat_data.to_csv(self.output_file_path_chat_level, index=False)
        self.user_data.to_csv(self.output_file_path_user_level, index=False)
        self.conv_data.to_csv(self.output_file_path_conv_level, index=False)