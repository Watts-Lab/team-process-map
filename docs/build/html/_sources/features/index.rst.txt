.. _features_technical:

Features: Technical Documentation
==================================

Below is a list of the features currently built and documented within our toolkit. We describe the different levels of analysis for features in :ref:`the Introduction, under Generating Features: Utterance-, Speaker-, and Conversation-Level <intro>`.

Utterance- (Chat) Level Features
*********************************
Utterance-Level features are calculated *first* in the Toolkit, as many conversation-Level features are derived from utterance-level information. These are the basic attributes that can be used to describe a single message ("utterance") in a conversation.

.. toctree::
   :maxdepth: 1

   basic_features
   certainty
   lexical_features_v2
   other_lexical_features
   info_exchange_zscore
   question_num
   politeness_features
   hedge
   temporal_features
   readability
   textblob_sentiment_analysis
   named_entity_recognition_features


Conversation-Level Features
****************************
Once utterance-level features are computed, we compute conversation-level features; some of these features represent an aggregation of utterance-level information (for example, the "average level of positivity" in a conversation is simply the mean positivity score for each utterance). Other conversation-level features are constructs that are defined only at the conversation-level, such as the level of "burstiness" in a team's communication patterns.

.. toctree::
   :maxdepth: 1

   burstiness
   information_diversity
   ../utils/gini_coefficient

Speaker- (User) Level Features
*********************************
User-level features currently represent an aggregation of features at the utterance- level (for example, the average number of words spoken *by a particular user*). There is therefore no separate speaker-level feature documentation; you may reference the :ref:`Speaker (User)-Level Features Page <user_level_features>` for more information.