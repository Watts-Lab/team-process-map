"""
file: featurize.py
---
This file is the main driver of the feature generating pipeline. 
It instantiates and calls the FeatureBuilder class which defines the logic used for feature creation.
"""

# Importing the Feature Generating Class
from feature_builder import FeatureBuilder

# Main Function
if __name__ == "__main__":
	
	
	# Instantiating the Feature Generating Class
	# Calling the "engine"/"driver" function of the FeatureBuilder class 
	# that creates the features, and writes them in output.
	# Defines one class for each dataset.

	# TINY Test sets --- just two conversations each
	# Tiny Juries
	# feature_builder = FeatureBuilder(
	# 	input_file_path = "../feature_engine/data/raw_data/juries_tiny_for_testing.csv",
	# 	output_file_path_chat_level = "../feature_engine/output/jury_TINY_output_chat_level.csv",
	# 	output_file_path_user_level = "../feature_engine/output/jury_TINY_output_user_level.csv",
	# 	output_file_path_conv_level = "../feature_engine/output/jury_TINY_output_conversation_level.csv",
	# )
	# feature_builder.featurize(col="message")

	# Tiny CSOP
	# tiny_csop_feature_builder = FeatureBuilder(
	# 	input_file_path = "../feature_engine/data/raw_data/csop_conversations_TINY.csv",
	# 	output_file_path_chat_level = "../feature_engine/output/csop_TINY_output_chat_level.csv",
	# 	output_file_path_user_level = "../feature_engine/output/csop_TINY_output_user_level.csv",
	# 	output_file_path_conv_level = "../feature_engine/output/csop_TINY_output_conversation_level.csv",
	# )
	# tiny_csop_feature_builder.featurize(col="message")

	#####

	# A tiny dataset built specifically for testing function and content word mimicry
	# mimicry_feature_builder = FeatureBuilder(
	# 	input_file_path = "../feature_engine/data/raw_data/test_mimicry.csv",
	# 	output_file_path_chat_level = "../feature_engine/output/test_mimicry_output_chat_level.csv",
	#	output_file_path_user_level = "../feature_engine/output/test_mimicry_output_user_level.csv",
	# 	output_file_path_conv_level = "../feature_engine/output/test_mimicry_output_conversation_level.csv"
	# )

	# mimicry_feature_builder.featurize(col="message")

	# A tiny dataset built specifically for testing NTRI
	# ntri_feature_builder = FeatureBuilder(
	# 	input_file_path = "../feature_engine/data/raw_data/test_ntri.csv",
	# 	output_file_path_chat_level = "../feature_engine/output/test_ntri_output_chat_level.csv",
	#	output_file_path_user_level = "../feature_engine/output/test_ntri_output_user_level.csv",
	# 	output_file_path_conv_level = "../feature_engine/output/test_ntri_output_conversation_level.csv"
	# )

	# ntri_feature_builder.featurize(col="message")

	# FULL DATASETS BELOW

	# Juries

	# jury_feature_builder_first25pct = FeatureBuilder(
	# 	input_file_path = "../feature_engine/data/raw_data/jury_conversations_with_outcome_var.csv",
	# 	output_file_path_chat_level = "../feature_engine/output/first_25/jury_output_chat_level.csv",
	# 	output_file_path_user_level = "../feature_engine/output/first_25/jury_output_user_level.csv",
	# 	output_file_path_conv_level = "../feature_engine/output/first_25/jury_output_conversation_level.csv",
	# 	analyze_first_pct = 0.25
	# )
	# jury_feature_builder_first25pct.featurize(col="message")

	# jury_feature_builder_first50pct = FeatureBuilder(
	# 	input_file_path = "../feature_engine/data/raw_data/jury_conversations_with_outcome_var.csv",
	# 	output_file_path_chat_level = "../feature_engine/output/first_50/jury_output_chat_level.csv",
	# 	output_file_path_user_level = "../feature_engine/output/first_50/jury_output_user_level.csv",
	# 	output_file_path_conv_level = "../feature_engine/output/first_50/jury_output_conversation_level.csv",
	# 	analyze_first_pct = 0.5
	# )
	# jury_feature_builder_first50pct.featurize(col="message")

	# jury_feature_builder_first75pct = FeatureBuilder(
	# 	input_file_path = "../feature_engine/data/raw_data/jury_conversations_with_outcome_var.csv",
	# 	output_file_path_chat_level = "../feature_engine/output/first_75/jury_output_chat_level.csv",
	# 	output_file_path_user_level = "../feature_engine/output/first_75/jury_output_user_level.csv",
	# 	output_file_path_conv_level = "../feature_engine/output/first_75/jury_output_conversation_level.csv",
	# 	analyze_first_pct = 0.75
	# )
	# jury_feature_builder_first75pct.featurize(col="message")

	# jury_feature_builder_first80pct = FeatureBuilder(
	# 	input_file_path = "../feature_engine/data/raw_data/jury_conversations_with_outcome_var.csv",
	# 	output_file_path_chat_level = "../feature_engine/output/first_80/jury_output_chat_level.csv",
	#	output_file_path_user_level = "../feature_engine/output/first_80/jury_output_user_level.csv",
	# 	output_file_path_conv_level = "../feature_engine/output/first_80/jury_output_conversation_level.csv",
	# 	analyze_first_pct = 0.8
	# )
	# jury_feature_builder_first80pct.featurize(col="message")

	# jury_feature_builder = FeatureBuilder(
	# 	input_file_path = "../feature_engine/data/raw_data/jury_conversations_with_outcome_var.csv",
	# 	output_file_path_chat_level = "../feature_engine/output/jury_output_chat_level.csv",
	# 	output_file_path_user_level = "../feature_engine/output/jury_output_user_level.csv",
	# 	output_file_path_conv_level = "../feature_engine/output/jury_output_conversation_level.csv"
	# )

	# jury_feature_builder.featurize(col="message")



	# CSOP (Abdullah)

	csop_feature_builder_first25pct = FeatureBuilder(
		input_file_path = "../feature_engine/data/raw_data/csop_conversations_withblanks.csv",
		output_file_path_chat_level = "../feature_engine/output/first_25/csop_output_chat_level.csv",
		output_file_path_user_level = "../feature_engine/output/first_25/csop_output_user_level.csv",
		output_file_path_conv_level = "../feature_engine/output/first_25/csop_output_conversation_level.csv",
		analyze_first_pct = 0.25
	)
	csop_feature_builder_first25pct.featurize(col="message")

	csop_feature_builder_first50pct = FeatureBuilder(
		input_file_path = "../feature_engine/data/raw_data/csop_conversations_withblanks.csv",
		output_file_path_chat_level = "../feature_engine/output/first_50/csop_output_chat_level.csv",
		output_file_path_user_level = "../feature_engine/output/first_50/csop_output_user_level.csv",
		output_file_path_conv_level = "../feature_engine/output/first_50/csop_output_conversation_level.csv",
		analyze_first_pct = 0.5
	)
	csop_feature_builder_first50pct.featurize(col="message")

	csop_feature_builder_first75pct = FeatureBuilder(
		input_file_path = "../feature_engine/data/raw_data/csop_conversations_withblanks.csv",
		output_file_path_chat_level = "../feature_engine/output/first_75/csop_output_chat_level.csv",
		output_file_path_user_level = "../feature_engine/output/first_75/csop_output_user_level.csv",
		output_file_path_conv_level = "../feature_engine/output/first_75/csop_output_conversation_level.csv",
		analyze_first_pct = 0.75
	)
	csop_feature_builder_first75pct.featurize(col="message")

	csop_feature_builder_first80pct = FeatureBuilder(
		input_file_path = "../feature_engine/data/raw_data/csop_conversations_withblanks.csv",
		output_file_path_chat_level = "../feature_engine/output/first_80/csop_output_chat_level.csv",
		output_file_path_user_level = "../feature_engine/output/first_80/csop_output_user_level.csv",
		output_file_path_conv_level = "../feature_engine/output/first_80/csop_output_conversation_level.csv",
		analyze_first_pct = 0.8
	)
	csop_feature_builder_first80pct.featurize(col="message")

	csop_feature_builder = FeatureBuilder(
		input_file_path = "../feature_engine/data/raw_data/csop_conversations_withblanks.csv",
		output_file_path_chat_level = "../feature_engine/output/csop_output_chat_level.csv",
		output_file_path_user_level = "../feature_engine/output/csop_output_user_level.csv",
		output_file_path_conv_level = "../feature_engine/output/csop_output_conversation_level.csv"
	)

	csop_feature_builder.featurize(col="message")

	

	# CSOP II (Nak Won Rim)

	# csopII_feature_builder_first25pct = FeatureBuilder(
	# 	input_file_path = "../feature_engine/data/raw_data/csopII_conversations_withblanks.csv",
	# 	output_file_path_chat_level = "../feature_engine/output/first_25/csopII_output_chat_level.csv",
	# 	output_file_path_user_level = "../feature_engine/output/first_25/csopII_output_user_level.csv",
	# 	output_file_path_conv_level = "../feature_engine/output/first_25/csopII_output_conversation_level.csv",
	# 	analyze_first_pct = 0.25
	# )
	# csopII_feature_builder_first25pct.featurize(col="message")

	# csopII_feature_builder_first50pct = FeatureBuilder(
	# 	input_file_path = "../feature_engine/data/raw_data/csopII_conversations_withblanks.csv",
	# 	output_file_path_chat_level = "../feature_engine/output/first_50/csopII_output_chat_level.csv",
	# 	output_file_path_user_level = "../feature_engine/output/first_50/csopII_output_user_level.csv",
	# 	output_file_path_conv_level = "../feature_engine/output/first_50/csopII_output_conversation_level.csv",
	# 	analyze_first_pct = 0.5
	# )
	# csopII_feature_builder_first50pct.featurize(col="message")

	# csopII_feature_builder_first75pct = FeatureBuilder(
	# 	input_file_path = "../feature_engine/data/raw_data/csopII_conversations_withblanks.csv",
	# 	output_file_path_chat_level = "../feature_engine/output/first_75/csopII_output_chat_level.csv",
	# 	output_file_path_user_level = "../feature_engine/output/first_75/csopII_output_user_level.csv",
	# 	output_file_path_conv_level = "../feature_engine/output/first_75/csopII_output_conversation_level.csv",
	# 	analyze_first_pct = 0.75
	# )
	# csopII_feature_builder_first75pct.featurize(col="message")

	# csopII_feature_builder_first80pct = FeatureBuilder(
	# 	input_file_path = "../feature_engine/data/raw_data/csopII_conversations_withblanks.csv",
	# 	output_file_path_chat_level = "../feature_engine/output/first_80/csopII_output_chat_level.csv",
	#	output_file_path_user_level = "../feature_engine/output/first_80/csopII_output_user_level.csv",
	# 	output_file_path_conv_level = "../feature_engine/output/first_80/csopII_output_conversation_level.csv",
	# analyze_first_pct = 0.8
	# )
	# csopII_feature_builder_first80pct.featurize(col="message")	


	# csopII_feature_builder = FeatureBuilder(
	# 	input_file_path = "../feature_engine/data/raw_data/csopII_conversations_withblanks.csv",
	# 	output_file_path_chat_level = "../feature_engine/output/csopII_output_chat_level.csv",
	# 	output_file_path_user_level = "../feature_engine/output/csopII_output_user_level.csv",
	# 	output_file_path_conv_level = "../feature_engine/output/csopII_output_conversation_level.csv"
	# )

	# csopII_feature_builder.featurize(col="message")

	

	# DAT - Divergent Association Task

	# dat_feature_builder_first25pct = FeatureBuilder(
	# 	input_file_path = "../feature_engine/data/raw_data/DAT_conversations_withblanks.csv",
	# 	output_file_path_chat_level = "../feature_engine/output/first_25/DAT_output_chat_level.csv",
	# 	output_file_path_user_level = "../feature_engine/output/first_25/DAT_output_user_level.csv",
	# 	output_file_path_conv_level = "../feature_engine/output/first_25/DAT_output_conversation_level.csv",
	# 	analyze_first_pct = 0.25
	# )
	# dat_feature_builder_first25pct.featurize(col="message")

	# dat_feature_builder_first50pct = FeatureBuilder(
	# 	input_file_path = "../feature_engine/data/raw_data/DAT_conversations_withblanks.csv",
	# 	output_file_path_chat_level = "../feature_engine/output/first_50/DAT_output_chat_level.csv",
	# 	output_file_path_user_level = "../feature_engine/output/first_50/DAT_output_user_level.csv",
	# 	output_file_path_conv_level = "../feature_engine/output/first_50/DAT_output_conversation_level.csv",
	# 	analyze_first_pct = 0.5
	# )
	# dat_feature_builder_first50pct.featurize(col="message")

	# dat_feature_builder_first75pct = FeatureBuilder(
	# 	input_file_path = "../feature_engine/data/raw_data/DAT_conversations_withblanks.csv",
	# 	output_file_path_chat_level = "../feature_engine/output/first_75/DAT_output_chat_level.csv",
	# 	output_file_path_user_level = "../feature_engine/output/first_75/DAT_output_user_level.csv",
	# 	output_file_path_conv_level = "../feature_engine/output/first_75/DAT_output_conversation_level.csv",
	# 	analyze_first_pct = 0.75
	# )
	# dat_feature_builder_first75pct.featurize(col="message")

	# dat_feature_builder_first80pct = FeatureBuilder(
	# 	input_file_path = "../feature_engine/data/raw_data/DAT_conversations_withblanks.csv",
	# 	output_file_path_chat_level = "../feature_engine/output/first_80/DAT_output_chat_level.csv",
	#	output_file_path_user_level = "../feature_engine/output/first_80/DAT_output_user_level.csv",
	# 	output_file_path_conv_level = "../feature_engine/output/first_80/DAT_output_conversation_level.csv",
	# 	analyze_first_pct = 0.8
	# )
	# dat_feature_builder_first80pct.featurize(col="message")

	# dat_feature_builder = FeatureBuilder(
	# 	input_file_path = "../feature_engine/data/raw_data/DAT_conversations_withblanks.csv",
	# 	output_file_path_chat_level = "../feature_engine/output/DAT_output_chat_level.csv",
	# 	output_file_path_user_level = "../feature_engine/output/DAT_output_user_level.csv",
	# 	output_file_path_conv_level = "../feature_engine/output/DAT_output_conversation_level.csv"
	# )
	# dat_feature_builder.featurize(col="message")


	# PGG (Small)
	# pgg_feature_builder = FeatureBuilder(
	# 	input_file_path = "../feature_engine/data/raw_data/pgg_conversations_withblanks.csv",
	# 	output_file_path_chat_level = "../feature_engine/output/pgg_output_chat_level.csv",
	#	output_file_path_user_level = "../feature_engine/output/pgg_output_user_level.csv",
	# 	output_file_path_conv_level = "../feature_engine/output/pgg_output_conversation_level.csv"
	# )
	# pgg_feature_builder.featurize(col="message")

	

	# Estimation (Gurcay)
	# gurcay_estimation_feature_builder_first25pct = FeatureBuilder(
	# 	input_file_path = "../feature_engine/data/raw_data/gurcay2015_group_estimation.csv",
	# 	output_file_path_chat_level = "../feature_engine/output/first_25/gurcay2015estimation_output_chat_level.csv",
	# 	output_file_path_user_level = "../feature_engine/output/first_25/gurcay2015estimation_output_user_level.csv",
	# 	output_file_path_conv_level = "../feature_engine/output/first_25/gurcay2015estimation_output_conversation_level.csv",
	# 	analyze_first_pct = 0.25
	# )
	# gurcay_estimation_feature_builder_first25pct.featurize(col="message")	

	# gurcay_estimation_feature_builder_first50pct = FeatureBuilder(
	# 	input_file_path = "../feature_engine/data/raw_data/gurcay2015_group_estimation.csv",
	# 	output_file_path_chat_level = "../feature_engine/output/first_50/gurcay2015estimation_output_chat_level.csv",
	# 	output_file_path_user_level = "../feature_engine/output/first_50/gurcay2015estimation_output_user_level.csv",
	# 	output_file_path_conv_level = "../feature_engine/output/first_50/gurcay2015estimation_output_conversation_level.csv",
	# 	analyze_first_pct = 0.50
	# )
	# gurcay_estimation_feature_builder_first50pct.featurize(col="message")

	# gurcay_estimation_feature_builder_first75pct = FeatureBuilder(
	# 	input_file_path = "../feature_engine/data/raw_data/gurcay2015_group_estimation.csv",
	# 	output_file_path_chat_level = "../feature_engine/output/first_75/gurcay2015estimation_output_chat_level.csv",
	# 	output_file_path_user_level = "../feature_engine/output/first_75/gurcay2015estimation_output_user_level.csv",
	# 	output_file_path_conv_level = "../feature_engine/output/first_75/gurcay2015estimation_output_conversation_level.csv",
	# 	analyze_first_pct = 0.75
	# )
	# gurcay_estimation_feature_builder_first75pct.featurize(col="message")	

	# gurcay_estimation_feature_builder_first80pct = FeatureBuilder(
	# 	input_file_path = "../feature_engine/data/raw_data/gurcay2015_group_estimation.csv",
	# 	output_file_path_chat_level = "../feature_engine/output/first_80/gurcay2015estimation_output_chat_level.csv",
	#	output_file_path_user_level = "../feature_engine/output/first_80/gurcay2015estimation_output_user_level.csv",
	# 	output_file_path_conv_level = "../feature_engine/output/first_80/gurcay2015estimation_output_conversation_level.csv",
	# 	analyze_first_pct = 0.80
	# )
	# gurcay_estimation_feature_builder_first80pct.featurize(col="message")

	# gurcay_estimation_feature_builder = FeatureBuilder(
	# 	input_file_path = "../feature_engine/data/raw_data/gurcay2015_group_estimation.csv",
	# 	output_file_path_chat_level = "../feature_engine/output/gurcay2015estimation_output_chat_level.csv",
	# 	output_file_path_user_level = "../feature_engine/output/gurcay2015estimation_output_user_level.csv",
	# 	output_file_path_conv_level = "../feature_engine/output/gurcay2015estimation_output_conversation_level.csv"
	# )
	# gurcay_estimation_feature_builder.featurize(col="message")

	# # # Estimation (Becker)
	# becker_estimation_feature_builder_first25pct = FeatureBuilder(
	# 	input_file_path = "../feature_engine/data/raw_data/becker_group_estimation.csv",
	# 	output_file_path_chat_level = "../feature_engine/output/first_25/beckerestimation_output_chat_level.csv",
	# 	output_file_path_user_level = "../feature_engine/output/first_25/beckerestimation_output_user_level.csv",
	# 	output_file_path_conv_level = "../feature_engine/output/first_25/beckerestimation_output_conversation_level.csv",
	# 	analyze_first_pct = 0.25
	# )
	# becker_estimation_feature_builder_first25pct.featurize(col="message")

	# becker_estimation_feature_builder_first50pct = FeatureBuilder(
	# 	input_file_path = "../feature_engine/data/raw_data/becker_group_estimation.csv",
	# 	output_file_path_chat_level = "../feature_engine/output/first_50/beckerestimation_output_chat_level.csv",
	# 	output_file_path_user_level = "../feature_engine/output/first_50/beckerestimation_output_user_level.csv",
	# 	output_file_path_conv_level = "../feature_engine/output/first_50/beckerestimation_output_conversation_level.csv",
	# 	analyze_first_pct = 0.5
	# )
	# becker_estimation_feature_builder_first50pct.featurize(col="message")

	# becker_estimation_feature_builder_first75pct = FeatureBuilder(
	# 	input_file_path = "../feature_engine/data/raw_data/becker_group_estimation.csv",
	# 	output_file_path_chat_level = "../feature_engine/output/first_75/beckerestimation_output_chat_level.csv",
	# 	output_file_path_user_level = "../feature_engine/output/first_75/beckerestimation_output_user_level.csv",
	# 	output_file_path_conv_level = "../feature_engine/output/first_75/beckerestimation_output_conversation_level.csv",
	# 	analyze_first_pct = 0.75
	# )
	# becker_estimation_feature_builder_first75pct.featurize(col="message")

	# becker_estimation_feature_builder_first80pct = FeatureBuilder(
	# 	input_file_path = "../feature_engine/data/raw_data/becker_group_estimation.csv",
	# 	output_file_path_chat_level = "../feature_engine/output/first_80/beckerestimation_output_chat_level.csv",
	# 	output_file_path_user_level = "../feature_engine/output/first_80/beckerestimation_output_user_level.csv",
	# 	output_file_path_conv_level = "../feature_engine/output/first_80/beckerestimation_output_conversation_level.csv",
	# 	analyze_first_pct = 0.8
	# )
	# becker_estimation_feature_builder_first80pct.featurize(col="message")

	# becker_estimation_feature_builder = FeatureBuilder(
	# 	input_file_path = "../feature_engine/data/raw_data/becker_group_estimation.csv",
	# 	output_file_path_chat_level = "../feature_engine/output/beckerestimation_output_chat_level.csv",
	# 	output_file_path_user_level = "../feature_engine/output/beckerestimation_output_user_level.csv",
	# 	output_file_path_conv_level = "../feature_engine/output/beckerestimation_output_conversation_level.csv"
	# )
	# becker_estimation_feature_builder.featurize(col="message")