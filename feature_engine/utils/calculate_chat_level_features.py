"""
file: calculate_chat_level_features.py
---
This file defines the ChatLevelFeaturesCalculator class using the modules defined in "features".
The intention behind this class is to use these modules and define any and all chat level features here. 

The steps needed to add a feature would be to:
- First define any building blocks that the feature would need in the appropriate "features" module (like word counter).
- Define a function within the class that uses these building blocks to build the feature and appends it 
  to the chat level dataframe as columns.
- Call the feature defining function in the driver function.
"""

# Importing modules from features
from features.politeness_features import *
from features.basic_features import *
from features.info_exchange_zscore import *
from features.lexical_features_v2 import *
from features.other_LIWC_features import *
from features.word_mimicry import *
from features.hedge import *
from features.textblob_sentiment_analysis import *
from features.readability import *
from features.positivity_zscore import *
from features.question_num import *
from features.temporal_features import *

# Importing utils
from utils.preload_word_lists import *
from utils.zscore_chats_and_conversation import get_zscore_across_all_chats, get_zscore_across_all_conversations

class ChatLevelFeaturesCalculator:
    def __init__(self, chat_data: pd.DataFrame, bert_sentiment_data: pd.DataFrame) -> None:
        """
            This function is used to initialize variables and objects that can be used by all functions of this class.

        PARAMETERS:
            @param chat_data (pd.DataFrame): This is a pandas dataframe of the chat level features read in from the input dataset.
        """
        self.chat_data = chat_data
        self.bert_sentiment_data = bert_sentiment_data # Load BERT 
        self.easy_dale_chall_words = get_dale_chall_easy_words() # load easy Dale-Chall words exactly once.
        self.function_words = get_function_words() # load function words exactly once
        self.question_words = get_question_words() # load question words exactly once
        
    def calculate_chat_level_features(self) -> pd.DataFrame:
        """
            This is the main driver function for this class.

        RETURNS:
            (pd.DataFrame): The chat level dataset given to this class during initialization along with 
                            new columns for each chat level feature.
        """

        # Concat sentiment BERT markers (done through preprocessing)
        self.concat_bert_features()
        
        # Text-Based Basic Features
        self.text_based_features()

        # lexical features
        self.lexical_features()

        # Other lexical features
        self.other_lexical_features()

        # Word Mimicry
        self.calculate_word_mimicry()

        # Hedge Features
        self.calculate_hedge_features()

        # TextBlob Sentiment features
        self.calculate_textblob_sentiment()

        # Info Exchange Feature
        self.info_exchange()
        
        # Positivity Z-Score
        self.positivity_zscore()

        # Dale-Chall readability features
        self.get_dale_chall_score_and_classfication()
        
         # Tempora; features
        self.get_temporal_features()

        # Politeness (ConvoKit)
        self.calculate_politeness_sentiment()

        # Return the input dataset with the chat level features appended (as columns)
        return self.chat_data
        
    def concat_bert_features(self) -> None:
        # concat bert features
        self.chat_data = pd.concat([self.chat_data, self.bert_sentiment_data], axis = 1)

    def text_based_features(self) -> None:
        """
            This function is used to implement the common text based featuers.
        """
        # Count Words
        self.chat_data["num_words"] = self.chat_data["message"].apply(count_words)
        
        # Count Characters
        self.chat_data["num_chars"] = self.chat_data["message"].apply(count_characters)
        
        # Count Messages        
        self.chat_data["num_messages"] = self.chat_data["message"].apply(count_messages)
        
    def info_exchange(self) -> None:
        """
            This function helps in extracting the different types of z-scores from the chats 
            (see features/info_exchange_zscore.py to learn more about how these features are calculated).
        """
        # Get Modified Wordcount: Total word count - first_singular pronouns
        self.chat_data["info_exchange_wordcount"] = get_info_exchange_wordcount(self.chat_data)
        
        # Get the z-score of each message across all chats
        self.chat_data["info_exchange_zscore_chats"] = get_zscore_across_all_chats(self.chat_data, "info_exchange_wordcount")

        # Get the z-score within each conversation
        self.chat_data["info_exchange_zscore_conversation"] = get_zscore_across_all_conversations(self.chat_data, "info_exchange_wordcount")

    def positivity_zscore(self) -> None:
        """
            This function calculates the z-score of a message's positivity
            Please see features/positivity_zscore.py to learn more about how these features are calculated.
        """
        self.chat_data["positivity_wordcount"] = get_positivity_wordcount(self.chat_data)
        
        # Get the z-score of each message across all chats
        self.chat_data["positivity_zscore_chats"] = get_zscore_across_all_chats(self.chat_data, "positivity_wordcount")

        # Get the z-score within each conversation
        self.chat_data["positivity_zscore_conversation"] = get_zscore_across_all_conversations(self.chat_data, "positivity_wordcount")

    def lexical_features(self) -> None:
        """
            This is a driver function that calls relevant functions in features/lexical_features.py to implement the lexical features.
        """
        self.chat_data = pd.concat([self.chat_data, liwc_features(self.chat_data)], axis = 1)
        
    def calculate_hedge_features(self) -> None:
        """
        This function helps to calculate features related to expressing hesitation (or 'hedge').
        """
        # Naive hedge (contains the word or not)
        self.chat_data["hedge_naive"] = self.chat_data["hedge_words"].apply(is_hedged_sentence_1)


    def calculate_textblob_sentiment(self) -> None:
        """
        This function helps to calculate features related to using TextBlob to return subjectivity and polarity.
        """
        self.chat_data["textblob_subjectivity"] = self.chat_data["message"].apply(get_subjectivity_score)
        self.chat_data["textblob_polarity"] = self.chat_data["message"].apply(get_polarity_score)


    def get_dale_chall_score_and_classfication(self) -> None:
        """
        This function helps to calculate the readability of a text according to its Dale-Chall score.
        """
        self.chat_data['dale_chall_score'] = self.chat_data['message'].apply(lambda x: dale_chall_helper(x, easy_words = self.easy_dale_chall_words))
        self.chat_data['dale_chall_classification'] = self.chat_data['dale_chall_score'].apply(classify_text_dalechall)

    def other_lexical_features(self) -> None:
        """
            This function extract the number of questions, classify whether the message contains clarification questions,
            calculate the word type-to-token ratio, and the proportion of first person pronouns from the chats
            (see features/other_LIWC_features.py to learn more about how these features are calculated)
        """
        # Get the number of questions in each message
        # naive: Number of Question Marks
        self.chat_data["num_question_naive"] = self.chat_data["message_lower_with_punc"].apply(lambda x: calculate_num_question_naive(x, question_words = self.question_words))
        # nltk: Using POS-tagging; commented out because Convokit/Politeness has a similar feature, and it's not clear this has an advantage?
        #self.chat_data["num_question_nltk"] = self.chat_data["message_lower_with_punc"].apply(lambda x: calculate_num_question_nltk(x))

        # Classify whether the message contains clarification questions
        self.chat_data["NTRI"] = self.chat_data["message_lower_with_punc"].apply(classify_NTRI)
        
        # Calculate the word type-to-token ratio
        self.chat_data["word_TTR"] = self.chat_data["message"].apply(get_word_TTR)
        
        # Calculate the proportion of first person pronouns from the chats
        self.chat_data["first_pronouns_proportion"] = get_proportion_first_pronouns(self.chat_data)
        
    def calculate_word_mimicry(self) -> None:
        """
            This function calculate the number of function words that also used in other’s prior turn,
            and the sum of inverse frequency of each content word that also occurred in the other’s immediately prior turn.
            (see features/word_mimicry.py to learn more about how these features are calculated)
            
            Note: this function takes the dataset WITHOUT any punctuations as input
        """

        # Extract function words / content words from a message
        self.chat_data["function_words"] = self.chat_data["message"].apply(lambda x: get_function_words_in_message(x, function_word_reference = self.function_words))
        self.chat_data["content_words"] = self.chat_data["message"].apply(lambda x: get_content_words_in_message(x, function_word_reference = self.function_words))
        
        # Extract the function words / content words that also appears in the immediate previous turn
        self.chat_data["function_word_mimicry"] = mimic_words(self.chat_data, "function_words")
        self.chat_data["content_word_mimicry"] = mimic_words(self.chat_data, "content_words")
        
        # Compute the number of function words that also appears in the immediate previous turn
        self.chat_data["function_word_accommodation"] = self.chat_data["function_word_mimicry"].apply(function_mimicry_score)
        
        # Compute the sum of inverse frequency of each content word that also occurred in the other’s immediately prior turn.
        self.chat_data["content_word_accommodation"] = Content_mimicry_score(self.chat_data, "content_words","content_word_mimicry")

        # Drop the function / content word columns -- we don't need them in the output
        self.chat_data = self.chat_data.drop(columns=['function_words', 'content_words', 'function_word_mimicry', 'content_word_mimicry'])

    def get_temporal_features(self) -> None:
        """
        Calculates features relevant to the timestamps of each chat.

        - time diff: The difference between messages sent.
        """
        if {'timestamp'}.issubset(self.chat_data.columns):
            self.chat_data["time_diff"] =  get_time_diff(self.chat_data,"timestamp") 

    def calculate_politeness_sentiment(self) -> None:
        """
            This function calls the Politeness module from Convokit and includes all outputted features.
        """
        transformed_df = self.chat_data['message'].apply(get_politeness_strategies).apply(pd.Series)
        transformed_df = transformed_df.rename(columns=lambda x: re.sub('^feature_politeness_==()','',x)[:-2].lower())

        # Concatenate the transformed dataframe with the original dataframe
        self.chat_data = pd.concat([self.chat_data, transformed_df], axis=1)
