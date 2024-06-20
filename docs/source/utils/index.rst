.. _utils:

Utilities
=========

The :ref:`FeatureBuilder <feature_builder>` relies on a number of utilities to help preprocess conversations and generate features. 

"Driver" Classes: Utterance-, Conversation-, and Speaker-Level Features
************************************************************************

The most important of these utilities are three classes --- the :ref:`ChatLevelFeaturesCalculator <chat_level_features>`, :ref:`ConversationLevelFeaturesCalculator <conv_level_features>`, and :ref:`UserLevelFeaturesCalculator <user_level_features>`. These classes "drive" the process of computing features for each utterance, or "chat"; each conversation; and each speaker, or "user". When you declare and run a FeatureBuilder, it automatically calls each of these classes to compute features for utterances, conversations, and speakers, respectively. Therefore, users *indirectly* interact with each of these classes by making the appropriate specifications in the FeatureBuilder. Users do not directly interact with these classes.

Additional documentation for the three core classes can be found on their respective pages.

.. toctree::
   :maxdepth: 1

   calculate_chat_level_features
   calculate_conversation_level_features
   calculate_user_level_features

Other Utilities
****************

The FeatureBuilder and its driver classes also rely on a number of other utilities, which help to load lexicons and word embeddings, apply preprocessing rules, and summarize features. These utilities are documented below, but should largely function without direct user interaction.

.. toctree::
   :maxdepth: 1

   preload_word_lists
   preprocess
   summarize_features
   zscore_chats_and_conversation
   assign_chunk_nums
   check_embeddings
   gini_coefficient