# Importing modules from features
from features.politeness_features import *
from features.basic_features import *
from features.info_exchange_zscore import *
from features.lexical_features_v2 import *
from features.other_lexical_features import *
from features.word_mimicry import *
from features.hedge import *
from features.textblob_sentiment_analysis import *
from features.readability import *
from features.question_num import *
from features.temporal_features import *
from features.fflow import *
from features.certainty import *
from features.politeness_v2 import *
from features.reddit_tags import *
from features.named_entity_recognition_features import*

# Importing utils
from utils.preload_word_lists import *
from utils.zscore_chats_and_conversation import get_zscore_across_all_chats, get_zscore_across_all_conversations

class ChatLevelFeaturesCalculator:
    """
    Initialize variables and objects used by the ChatLevelFeaturesCalculator class.

    This class uses various feature modules to define chat-level features. It reads input data and
    initializes variables required to compute the features.

    :param chat_data: Pandas dataframe of chat-level features read from the input dataset
    :type chat_data: pd.DataFrame
    :param vect_data: Pandas dataframe containing vector data
    :type vect_data: pd.DataFrame
    :param bert_sentiment_data: Pandas dataframe containing BERT sentiment data
    :type bert_sentiment_data: pd.DataFrame
    :param ner_training_df: This is a pandas dataframe of training data for named entity recognition feature
    :type ner_training_df: pd.DataFrame
    :param ner_cutoff: This is the cutoff value for the confidence of prediction for each named entity
    :type ner_cutoff: int
    """
    def __init__(self, chat_data: pd.DataFrame, 
         vect_data: pd.DataFrame, 
         bert_sentiment_data: pd.DataFrame, 
         ner_training: pd.DataFrame,
         ner_cutoff: int) -> None:

        self.chat_data = chat_data
        self.vect_data = vect_data
        self.bert_sentiment_data = bert_sentiment_data # Load BERT 
        self.easy_dale_chall_words = get_dale_chall_easy_words() # load easy Dale-Chall words exactly once.
        self.function_words = get_function_words() # load function words exactly once
        self.question_words = get_question_words() # load question words exactly once
        self.first_person = get_first_person_words() # load first person words exactly once

        self.ner_training = ner_training
        self.ner_cutoff = ner_cutoff
        
    def calculate_chat_level_features(self) -> pd.DataFrame:
        """
        Main driver function for creating chat-level features.

        This function computes various chat-level features using different modules and appends 
        them as new columns to the input chat data.

        :return: The chat-level dataset with new columns for each chat-level feature
        :rtype: pd.DataFrame
        """

        # NER
        self.get_named_entity()

        # Concat sentiment BERT markers (done through preprocessing)
        self.concat_bert_features()
        
        # Text-Based Basic Features
        self.text_based_features()

        # "Basic" Info Exchange Feature -- z-scores of content minus first pronouns
        self.info_exchange()

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
        
        # Positivity Z-Score
        self.positivity_zscore()

        # Dale-Chall readability features
        self.get_dale_chall_score_and_classfication()
        
        # Temporal features
        self.get_temporal_features()

        # Politeness (ConvoKit)
        self.calculate_politeness_sentiment()

        # Politeness (Yeomans/Bevis)
        self.calculate_politeness_v2()

        # Forward Flow
        self.get_forward_flow()

        # Certainty
        self.get_certainty_score()

        # Reddit / online communication features
        self.get_reddit_features()

        # Return the input dataset with the chat level features appended (as columns)
        return self.chat_data
        
    def concat_bert_features(self) -> None:
        """
        Concatenate RoBERTa sentiment features to the chat data.

        This function appends RoBERTa sentiment data (which are pre-processed beforehand to save computation)
        as new columns to the existing chat data.

        :return: None
        :rtype: None
        """
        self.chat_data = pd.concat([self.chat_data, self.bert_sentiment_data], axis = 1)

    def text_based_features(self) -> None:
        """
        Implement common text-based features.

        This function calculates and appends the following text-based features to the chat data:
        - Number of words
        - Number of characters
        - Number of messages

        :return: None
        :rtype: None
        """
        # Count Words
        self.chat_data["num_words"] = self.chat_data["message"].apply(count_words)
        
        # Count Characters
        self.chat_data["num_chars"] = self.chat_data["message"].apply(count_characters)
        
        # Count Messages        
        self.chat_data["num_messages"] = self.chat_data["message"].apply(count_messages)
        
    def info_exchange(self) -> None:
        """
        Extract different types of z-scores from the chats.

        This function calculates and appends the following info exchange features to the chat data:
        - Modified word count (total word count minus first singular pronouns)
        - Z-score of the modified word count across all chats
        - Z-score of the modified word count within each conversation

        It then drops the intermediate `info_exchange_wordcount` column as it is a linear combination of the z-score columns.

        :return: None
        :rtype: None
        """

        # Get Modified Wordcount: Total word count - first_singular pronouns
        self.chat_data["info_exchange_wordcount"] = get_info_exchange_wordcount(self.chat_data, self.first_person)
        
        # Get the z-score of each message across all chats
        self.chat_data["info_exchange_zscore_chats"] = get_zscore_across_all_chats(self.chat_data, "info_exchange_wordcount")

        # Get the z-score within each conversation
        self.chat_data["info_exchange_zscore_conversation"] = get_zscore_across_all_conversations(self.chat_data, "info_exchange_wordcount")

        # Drop the info exchange wordcount --- it's a linear combination of 2 columns and therefore useless
        self.chat_data = self.chat_data.drop(columns=['info_exchange_wordcount'])

    def positivity_zscore(self) -> None:
        """
        Calculate the z-score of a message's positivity (as measured by RoBERTa).

        This function calculates and appends the following positivity z-score features to the chat data:
        - Z-score of the positivity across all chats
        - Z-score of the positivity within each conversation

        :return: None
        :rtype: None
        """
        # Get the z-score of each message across all chats
        self.chat_data["positivity_zscore_chats"] = get_zscore_across_all_chats(self.chat_data, "positive_bert")

        # Get the z-score within each conversation
        self.chat_data["positivity_zscore_conversation"] = get_zscore_across_all_conversations(self.chat_data, "positive_bert")

    def lexical_features(self) -> None:
        """
        Implement lexical features.

        This driver function calls relevant functions to compute lexical features and appends them to the chat data.

        :return: None
        :rtype: None
        """
        self.chat_data = pd.concat([self.chat_data, liwc_features(self.chat_data)], axis = 1)
        
    def calculate_hedge_features(self) -> None:
        """
        Calculate features related to expressing hesitation (or 'hedge').

        This function identifies whether a message contains hedge words using a naive approach and appends this 
        information as a new column to the chat data.

        :return: None
        :rtype: None
        """
        # Naive hedge (contains the word or not)
        self.chat_data["hedge_naive"] = self.chat_data["hedge_words_lexical_per_100"].apply(is_hedged_sentence_1)

    def calculate_textblob_sentiment(self) -> None:
        """
        Calculate features related to sentiment using TextBlob.

        This function calculates and appends the following TextBlob sentiment features to the chat data:
        - Subjectivity score
        - Polarity score

        :return: None
        :rtype: None
        """
        self.chat_data["textblob_subjectivity"] = self.chat_data["message"].apply(get_subjectivity_score)
        self.chat_data["textblob_polarity"] = self.chat_data["message"].apply(get_polarity_score)

    def get_dale_chall_score_and_classfication(self) -> None:
        """
        Calculate the readability of a text according to its Dale-Chall score.

        This function calculates and appends the following Dale-Chall readability features to the chat data:
        - Dale-Chall score
        - Dale-Chall classification

        :return: None
        :rtype: None
        """
        self.chat_data['dale_chall_score'] = self.chat_data['message'].apply(lambda x: dale_chall_helper(x, easy_words = self.easy_dale_chall_words))
        self.chat_data['dale_chall_classification'] = self.chat_data['dale_chall_score'].apply(classify_text_dalechall)

    def other_lexical_features(self) -> None:
        """
        Extract various lexical features from the chats.

        This function calculates and appends the following lexical features to the chat data:
        - Number of questions (naive approach using question marks and question words)
        - Classification of whether the message contains clarification questions
        - Word type-to-token ratio (TTR)
        - Proportion of first-person pronouns

        It also drops the raw number of first-person pronouns from the chat data as it is proportional to other columns.

        :return: None
        :rtype: None
        """

        # Get the number of questions in each message
        # naive: Number of Question Marks + Sentences that start with question words
        self.chat_data["num_question_naive"] = self.chat_data["message_lower_with_punc"].apply(lambda x: calculate_num_question_naive(x, question_words = self.question_words))
        
        # Classify whether the message contains clarification questions
        self.chat_data["NTRI"] = self.chat_data["message_lower_with_punc"].apply(classify_NTRI)
        
        # Calculate the word type-to-token ratio
        self.chat_data["word_TTR"] = self.chat_data["message"].apply(get_word_TTR)
        
        # Calculate the proportion of first person pronouns from the chats
        self.chat_data["first_pronouns_proportion"] = get_proportion_first_pronouns(self.chat_data)

        # drop the raw number of first pronouns -- unnecessary given this is proportional to other first-pronoun columns
        self.chat_data = self.chat_data.drop(columns=['first_person_raw'])
   
    def calculate_word_mimicry(self) -> None:
        """
        Calculate features related to word mimicry.

        This function calculates the number of function words and the sum of inverse frequency of content words
        that also appear in the other’s prior turn. It also computes mimicry using SBERT vectors.

        - Extracts function and content words from a message
        - Identifies mimicry of function and content words from the immediate previous turn
        - Computes function word accommodation (number of mimicked function words)
        - Computes content word accommodation (sum of inverse frequency of mimicked content words)
        - Computes mimicry relative to the previous chat or messages using SBERT vectors

        Drops the intermediate columns related to function and content words after calculation.

        :return: None
        :rtype: None
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

        # Compute the mimicry relative to the previous chat(s) using SBERT vectors
        self.chat_data["mimicry_bert"] = get_mimicry_bert(self.chat_data, self.vect_data)
        self.chat_data["moving_mimicry"] = get_moving_mimicry(self.chat_data, self.vect_data)
        
    def get_temporal_features(self) -> None:
        """
        Calculate features relevant to the timestamps of each chat.

        This function calculates and appends the following temporal feature to the chat data:
        - Time difference between messages sent
        
        It checks whether the 'timestamp' column is available. If not, it tries to calculate 
        using 'timestamp_start' and 'timestamp_end' columns.

        :return: None
        :rtype: None
        """
        if {'timestamp'}.issubset(self.chat_data.columns):
            self.chat_data["time_diff"] =  get_time_diff(self.chat_data,"timestamp") 
        elif {'timestamp_start', 'timestamp_end'}.issubset(self.chat_data.columns):
            self.chat_data["time_diff"] =  get_time_diff_startend(self.chat_data)
        if {'timestamp'}.issubset(self.chat_data.columns):
            self.chat_data["time_diff"] =  get_time_diff(self.chat_data,"timestamp") 
        elif {'timestamp_start', 'timestamp_end'}.issubset(self.chat_data.columns):
            self.chat_data["time_diff"] =  get_time_diff_startend(self.chat_data)

    def calculate_politeness_sentiment(self) -> None:
        """
        Calculate politeness strategies using the Politeness module from Convokit.

        This function applies the Convokit politeness strategies to the chat messages and 
        appends all outputted features to the chat data.

        :return: None
        :rtype: None
        """
        transformed_df = self.chat_data['message_lower_with_punc'].apply(get_politeness_strategies).apply(pd.Series)
        transformed_df = transformed_df.rename(columns=lambda x: re.sub('^feature_politeness_==()','',x)[:-2].lower())

        # Concatenate the transformed dataframe with the original dataframe
        self.chat_data = pd.concat([self.chat_data, transformed_df], axis=1)

    def calculate_politeness_v2(self) -> None:
        """
        Calculate politeness features using the System for Encouraging Conversational Receptiveness (SECR).

        Source: https://www.mikeyeomans.info/papers/receptiveness.pdf

        This function applies the SECR module to the chat messages and appends the calculated 
        politeness features to the chat data.

        :return: None
        :rtype: None
        """
        self.chat_data = pd.concat([self.chat_data, get_politeness_v2(self.chat_data, 'message_lower_with_punc')], axis=1) 

    def get_forward_flow(self) -> None:
        """
        Calculate the chat-level forward flow.

        This function compares the current chat to the average of the previous chats 
        and appends the forward flow score to the chat data.

        :return: None
        :rtype: None
        """
        self.chat_data["forward_flow"] = get_forward_flow(self.chat_data, self.vect_data)
   
    def get_certainty_score(self) -> None:
        """
        Calculate the certainty score of a statement.

        This function uses the formula published in Rocklage et al. (2023) to calculate 
        the certainty score of a chat message and appends it to the chat data.

        Source: https://journals.sagepub.com/doi/pdf/10.1177/00222437221134802

        :return: None
        :rtype: None
        """
        self.chat_data["certainty_rocklage"] = self.chat_data["message_lower_with_punc"].apply(get_certainty)

    def get_reddit_features(self) -> None:
        """
        Calculate a suite of features common in online communication.

        This function calculates and appends the following features to the chat data:
        - Number of all caps words
        - Number of links
        - Number of user references (Reddit format)
        - Number of emphases (bold, italics)
        - Number of bullet points
        - Number of numbered points
        - Number of line breaks
        - Number of quotes
        - Number of responses to someone else (using ">")
        - Number of ellipses
        - Number of parentheses
        - Number of emojis

        :return: None
        :rtype: None
        """
        self.chat_data["num_all_caps"] = self.chat_data["message_original"].apply(count_all_caps)
        self.chat_data["num_links"] = self.chat_data["message_lower_with_punc"].apply(count_links)
        self.chat_data["num_reddit_users"] = self.chat_data["message_lower_with_punc"].apply(count_user_references)
        self.chat_data["num_emphasis"] = self.chat_data["message_lower_with_punc"].apply(count_emphasis)
        self.chat_data["num_bullet_points"] = self.chat_data["message_lower_with_punc"].apply(count_bullet_points)
        self.chat_data["num_numbered_points"] = self.chat_data["message_lower_with_punc"].apply(count_numbering)
        self.chat_data["num_line_breaks"] = self.chat_data["message_lower_with_punc"].apply(count_line_breaks)
        self.chat_data["num_quotes"] = self.chat_data["message_lower_with_punc"].apply(count_quotes)
        self.chat_data["num_block_quote_responses"] = self.chat_data["message_lower_with_punc"].apply(count_responding_to_someone)
        self.chat_data["num_ellipses"] = self.chat_data["message_lower_with_punc"].apply(count_ellipses)
        self.chat_data["num_parentheses"] = self.chat_data["message_lower_with_punc"].apply(count_parentheses)
        self.chat_data["num_emoji"] = self.chat_data["message_lower_with_punc"].apply(count_emojis)
    
    def get_named_entity(self) -> None:
        """
        This function calculates the number of named entities in a chat.
        
        :return: None
        :rtype: None
        """

        if self.ner_training is not None:
            train_spacy_ner(self.ner_training)
            self.chat_data["num_named_entity"] = self.chat_data["message"].apply(num_named_entity, cutoff=self.ner_cutoff)
            self.chat_data["named_entities"] = self.chat_data["message"].apply(named_entities, cutoff=self.ner_cutoff)
