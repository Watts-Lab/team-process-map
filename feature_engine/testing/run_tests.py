"""
file: run_tests.py
---
This file runs the feature builder pipeline on our custom testing datasets.
"""

# Importing the Feature Generating Class
import sys
import os

# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now you can import feature_builder
from feature_builder import FeatureBuilder
import pandas as pd
import chardet

# Main Function
if __name__ == "__main__":

	# detects CSV encoding of our datasets
	with open("data/cleaned_data/test_chat_level.csv", 'rb') as file:
		chat_encoding = chardet.detect(file.read())

	with open("data/cleaned_data/test_conv_level.csv", 'rb') as file:
		conv_encoding = chardet.detect(file.read())

	chat_df = pd.read_csv("data/cleaned_data/test_chat_level.csv", encoding=chat_encoding['encoding'])
	conv_df = pd.read_csv("data/cleaned_data/test_conv_level.csv", encoding=conv_encoding['encoding'])
	test_ner_df = pd.read_csv("data/cleaned_data/test_named_entity.csv", encoding='utf-8')
	test_ner_training_df = pd.read_csv("data/cleaned_data/train_named_entity.csv")
	chat_complex_df = pd.read_csv("data/cleaned_data/test_chat_level_complex.csv", encoding=chat_encoding['encoding'])
	conv_complex_df = pd.read_csv("data/cleaned_data/test_conv_level_complex.csv", encoding=chat_encoding['encoding'])
		
	# TESTING DATASETS -------------------------------

	testing_chat = FeatureBuilder(
		input_df = chat_df,
		vector_directory = "../tpm-data/vector_data/",
		output_file_path_chat_level = "../output/chat/test_chat_level_chat.csv",
		output_file_path_user_level = "../output/user/test_chat_level_user.csv",
		output_file_path_conv_level = "../output/conv/test_chat_level_conv.csv",
		turns = False,
	)
	testing_chat.featurize(col="message")

	# testing_conv = FeatureBuilder(
	# 	input_df = conv_df,
	# 	vector_directory = "../tpm-data/vector_data/",
	# 	output_file_path_chat_level = "../output/chat/test_conv_level_chat.csv",
	# 	output_file_path_user_level = "../output/user/test_conv_level_user.csv",
	# 	output_file_path_conv_level = "../output/conv/test_conv_level_conv.csv",
	# 	turns = False,
	# )
	# testing_conv.featurize(col="message")

	# test_ner_feature_builder = FeatureBuilder(
	# 	input_df = test_ner_df,
	# 	ner_training_df = test_ner_training_df,
	# 	vector_directory = "../tpm-data/vector_data/",
	# 	output_file_path_chat_level = "../output/chat/test_named_entity_chat_level.csv",
	# 	output_file_path_user_level = "../output/user/test_named_entity_user_level.csv",
	# 	output_file_path_conv_level = "../output/conv/test_named_entity_conversation_level.csv",
	# 	turns = False,
	# 	conversation_id = "stageId",
	# 	cumulative_grouping = True
	# )
	# test_ner_feature_builder.featurize(col="message")

	# # testing perturbed chat level features
	# testing_chat_complex = FeatureBuilder(
	# 	input_df = chat_complex_df,
	# 	vector_directory = "../tpm-data/vector_data/",
	# 	output_file_path_chat_level = "../output/chat/test_chat_level_chat_complex.csv",
	# 	output_file_path_user_level = "../output/user/test_chat_level_user_complex.csv",
	# 	output_file_path_conv_level = "../output/conv/test_chat_level_conv_complex.csv",
	# 	turns = False,
	# )
	# testing_chat_complex.featurize(col="message")

	# # testing conv features
	# testing_conv_complex = FeatureBuilder(
	# 	input_df = conv_complex_df,
	# 	vector_directory = "../feature_engine/tpm-data/vector_data/",
	# 	output_file_path_chat_level = "../output/chat/test_conv_level_chat_complex.csv",
	# 	output_file_path_user_level = "../output/user/test_conv_level_user_complex.csv",
	# 	output_file_path_conv_level = "../output/conv/test_conv_level_conv_complex.csv",
	# 	turns = False,
	# )
	# testing_conv_complex.featurize(col="message")