"""
file: featurize.py
---
This file is the main driver of the feature generating pipeline. 
It instantiates and calls the FeatureBuilder class which defines the logic used for feature creation.
"""

# Importing the Feature Generating Class
from feature_builder import FeatureBuilder
import pandas as pd
import chardet

# Main Function
if __name__ == "__main__":

	# our test datasets
	tiny_juries_df = pd.read_csv("../feature_engine/tpm-data/cleaned_data/test_data/juries_tiny_for_testing.csv", encoding='utf-8')
	tiny_multi_task_df = pd.read_csv("../feature_engine/tpm-data/cleaned_data/test_data/multi_task_TINY.csv", encoding='utf-8')
	juries_df = pd.read_csv("../feature_engine/tpm-data/cleaned_data/jury_conversations_with_outcome_var.csv", encoding='utf-8')
	csop_df = pd.read_csv("../feature_engine/tpm-data/cleaned_data/csop_conversations_withblanks.csv", encoding='utf-8')
	csopII_df = pd.read_csv("../feature_engine/tpm-data/cleaned_data/csopII_conversations_withblanks.csv", encoding='utf-8')
	
	# TINY / TEST DATASETS -------------------------------#
	
	# # Tiny Juries
	# tiny_juries_feature_builder = FeatureBuilder(
	# 	input_df = tiny_juries_df,
	# 	vector_directory = "../feature_engine/tpm-data/vector_data/",
	# 	output_file_path_chat_level = "../feature_engine/output/chat/jury_TINY_output_chat_level.csv",
	# 	output_file_path_user_level = "../feature_engine/output/user/jury_TINY_output_user_level.csv",
	# 	output_file_path_conv_level = "../feature_engine/output/conv/jury_TINY_output_conversation_level.csv",
	# 	turns = False,
	# )
	# feature_builder.featurize(col="message")

	# # Tiny multi-task
	# tiny_multi_task_feature_builder = FeatureBuilder(
	# 	input_df = tiny_multi_task_df,
	# 	vector_directory = "../feature_engine/tpm-data/vector_data/",
	# 	output_file_path_chat_level = "../feature_engine/output/chat/multi_task_TINY_output_chat_level_stageId_cumulative.csv",
	# 	output_file_path_user_level = "../feature_engine/output/user/multi_task_TINY_output_user_level_stageId_cumulative.csv",
	# 	output_file_path_conv_level = "../feature_engine/output/conv/multi_task_TINY_output_conversation_level_stageId_cumulative.csv",
	# 	turns = False,
	# 	conversation_id = "stageId",
	# 	cumulative_grouping = True
	# )
	# tiny_multi_task_feature_builder.featurize(col="message")

	# FULL DATASETS BELOW ------------------------------------- #
	
	# Juries
	# jury_feature_builder = FeatureBuilder(
	# 	input_df = juries_df,
	# 	vector_directory = "../feature_engine/tpm-data/vector_data/",
	# 	output_file_path_chat_level = "../feature_engine/output/chat/jury_output_chat_level.csv",
	# 	# output_file_path_chat_level = "",
	# 	output_file_path_user_level = "../feature_engine/output/user/jury_output_user_level.csv",
	# 	output_file_path_conv_level = "../feature_engine/output/conv/jury_output_conversation_level.csv",
	# 	turns = True
	# )
	# jury_feature_builder.featurize(col="message")

	# # CSOP (Abdullah)
	# csop_feature_builder = FeatureBuilder(
	# 	input_df = csop_df,
	# 	vector_directory = "../feature_engine/tpm-data/vector_data/",
	# 	output_file_path_chat_level = "../feature_engine/output/chat/csop_output_chat_level.csv",
	# 	output_file_path_user_level = "../feature_engine/output/user/csop_output_user_level.csv",
	# 	output_file_path_conv_level = "../feature_engine/output/conv/csop_output_conversation_level.csv",
	# 	turns = True
	# )
	# csop_feature_builder.featurize(col="message")


	# # CSOP II (Nak Won Rim)
	# csopII_feature_builder = FeatureBuilder(
	# 	input_df = csopII_df,
	# 	vector_directory = "../feature_engine/tpm-data/vector_data/",
	# 	output_file_path_chat_level = "../feature_engine/output/chat/csopII_output_chat_level.csv",
	# 	output_file_path_user_level = "../feature_engine/output/user/csopII_output_user_level.csv",
	# 	output_file_path_conv_level = "../feature_engine/output/conv/csopII_output_conversation_level.csv",
	# 	turns = True
	# )
	# csopII_feature_builder.featurize(col="message")
