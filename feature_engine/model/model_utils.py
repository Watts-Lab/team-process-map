from model_builder import ModelBuilder
import random
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import colorsys
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from bokeh.plotting import figure, output_file, show
from bokeh.models import HoverTool, ColumnDataSource, ColorMapper
from bokeh.transform import linear_cmap
from bokeh.palettes import Viridis256
from itertools import cycle
plt.rcParams["font.family"] = "Times New Roman"

def repeated_kfold_cv(model, k = 10, seed = 19104):

	"""
	Parameters:
	- model: The model we are doing k-fold CV for
	- k: the number of fols (defaults to 10)
	- seed: the random seed (defaults to 19104)

	@returns the following, pouplated with data from the k0=-fold CV:
	- train_metrics: a dataframe to store all the training metrics
	- test_metrics: a dataframe to store all the test set metrics (we will universally use a 80-20 train-test split)
	- shap_df: a dataframe to store the Shapley value summaries for each fold
	- shap_correlation_df: a dataframe to store how the Shapley values correlate with feature values for each fold
	"""

	# Repeated k-fold cross-validation
	random.seed(seed) # set seed for reproducibility
	random_states_list = [random.randint(100, 1000000) for _ in range(k)] # create a bunch of different random states

	# Store metrics --- R^2, MAE, MSE
	metrics = ['r2', 'mae', 'mse', 'rmse']
	train_metrics = pd.DataFrame(columns=metrics)
	test_metrics = pd.DataFrame(columns=metrics)

	for i in range(len(random_states_list)):
		# store the model metrics for each iteration
		metrics = model.evaluate_model(model.baseline_model, val_size = 0.2, test_size = None, random_state = random_states_list[i], visualize_model = False)        
		train_metrics = pd.concat([train_metrics, pd.DataFrame(metrics['train'], columns=metrics)], ignore_index=True)
		test_metrics = pd.concat([test_metrics, pd.DataFrame(metrics['val'], columns=metrics)], ignore_index=True)

		# store the shap summary for each iteration
		try:     
			shap_summary = model.shap_summary
			shap_df = pd.merge(shap_df, shap_summary[['feature', 'shap']], on='feature')
			shap_df.rename(columns={'shap': f'shap_{i+1}'}, inplace=True)
			shap_correlation_df = pd.merge(shap_correlation_df, shap_summary[['feature', 'correlation_btw_shap_and_feature_value']], on='feature')
			shap_correlation_df.rename(columns={'correlation_btw_shap_and_feature_value': f'cor_{i+1}'}, inplace=True)
		except NameError:
			# we haven't defined these yet; we're in the first iteration!
			# we have to do this becaus model.X does not show up until after the first case when evaluate_model is called
			shap_df = pd.DataFrame({'feature': model.X.columns})
			shap_correlation_df = pd.DataFrame({'feature': model.X.columns})

			shap_summary = model.shap_summary
			shap_df = pd.merge(shap_df, shap_summary[['feature', 'shap']], on='feature')
			shap_df.rename(columns={'shap': f'shap_{i+1}'}, inplace=True)
			shap_correlation_df = pd.merge(shap_correlation_df, shap_summary[['feature', 'correlation_btw_shap_and_feature_value']], on='feature')
			shap_correlation_df.rename(columns={'correlation_btw_shap_and_feature_value': f'cor_{i+1}'}, inplace=True)

	shap_df.set_index('feature', inplace=True)
	shap_correlation_df.set_index('feature', inplace=True)

	return(shap_df, shap_correlation_df, train_metrics, test_metrics)

def get_repeated_kfold_cv_summary(shap_df, shap_correlation_df, train_metrics, test_metrics):
	"""
	Get the means of the repeated k-fold cross validation across all relevant metrics.
	"""
	shap_means = shap_df.mean(axis=1).sort_values(ascending = False)
	shap_cor_means = shap_correlation_df.mean(axis=1).reindex(index = shap_means.index)
	train_means = train_metrics.mean()
	test_means = test_metrics.mean()

	return(shap_means, shap_cor_means, train_means, test_means)


def plot_important_features_over_time(merged_df, color_palette, title="Top Feature Importance Over Time", filename="./figures/feature_importance.png"):
	# Transpose the DataFrame and sort by each time point
	top_features = merged_df.apply(lambda x: x.nlargest(NUM_TOP_FEATURES))

	non_na_feature = top_features.dropna()
	na_feature = top_features.loc[~top_features.index.isin(top_features.dropna().index)].fillna(0)

	# Plot a line chart to show how the top feature values change over time
	if not non_na_feature.empty and not na_feature.empty:
		ax = non_na_feature.T.plot(kind='line', marker='o', linewidth=3, color=color_palette)
		na_feature.T.plot(kind='line', marker='o', linestyle='--', linewidth=3, ax=ax, color=color_palette)  # Use the same axis for dashed lines
	elif non_na_feature.empty:
		ax = na_feature.T.plot(kind='line', marker='o', linestyle='--', linewidth=3, color=color_palette)
	elif na_feature.empty:
		ax = non_na_feature.T.plot(kind='line', marker='o', linewidth=3, color=color_palette)

	plt.ylabel('Importance (SHAP value)', size=14)
	plt.xlabel('Percent of Chat Messages (Chronological)', size=14)
	plt.title(title, fontsize=18, fontweight="bold")
	plt.xticks(range(len(time_points)), time_points, fontsize=14)

	# Update legend with custom color mapping
	legend_labels = ax.get_legend().get_texts()
	for label in legend_labels:
		feature_name = label.get_text()

	plt.legend(loc='upper left', fontsize=12, bbox_to_anchor=(1.05, 1), bbox_transform=ax.transAxes)

	plt.savefig(filename, dpi=1200, bbox_inches='tight')
	plt.show()

def plot_r2_and_mse_over_time(metrics, title):
	"""

	"""

	# Transpose the data for plotting
	transposed_data = metrics.T
	fig, ax1 = plt.subplots()

	plt.style.use({"figure.facecolor": "white", "axes.facecolor": "white"})

	# Create the left y-axis for R^2
	ax1.set_ylabel("R^2", fontsize=14)
	ax1.plot(time_points, transposed_data["r2"], label="R^2", color="cadetblue", marker="o", linewidth=3)
	ax1.tick_params(axis="y", size=14)

	# Create the right y-axis for MSE
	ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
	ax2.set_ylabel("MSE", fontsize=14)
	ax2.plot(time_points, transposed_data["mse"], label="MSE", color="mediumorchid", marker="o", linewidth=3)
	ax2.tick_params(axis="y", size=14)

	#x-axis font size
	ax1.tick_params(axis="x", labelsize=14)
	ax1.set_xlabel('Percent of Chat Messages (Chronological)', size=14)

	# Combine the legends for both lines
	lines, labels = ax1.get_legend_handles_labels()
	lines2, labels2 = ax2.get_legend_handles_labels()
	ax2.legend(lines + lines2, labels + labels2, loc="upper left")

	# Adjust layout
	plt.title(title, fontweight="bold", fontsize=18)
	plt.tight_layout()
	plt.show()


def check_na_rows_cols(df):
	"""
	Check for NaN values in columns.

	@param df: the dataframe being checked.

	Returns: Tuple of (nan_columns, nan_rows)
	- List of columns that contain NA
	- List of rows that contain NA
	"""
	nan_columns = df.columns[df.isna().any()].tolist()

	# Check for NaN values in rows
	nan_rows = df.index[df.isna().any(axis=1)].tolist()

	# Display the columns and rows with NaN values
	print("Columns with NaN values:", nan_columns)
	print("Rows with NaN values:", nan_rows)

	return(nan_columns, nan_rows)

'''
Functions for Plotting the Principal Components
'''
def get_numeric_cols(dfs):
	"""
	Filters dataframes to ensure that all columns are numeric

	@param dfs: list of dataframes
	"""
	new_dfs=[]
	for df in dfs:
		new_dfs.append(df.select_dtypes(['number']))
	return new_dfs

def drop_non_common_columns(dfs):
	"""
	Drops any columns that are not shared between dataframes,
	so that they can be plotted on the same axis.

	@param dfs: list of dataframes
	"""

	# Check if there are any data frames in the list
	if not dfs:
		return []

	# Find the intersection of columns in all data frames
	common_columns = set(dfs[0].columns)
	for df in dfs[1:]:
		common_columns = common_columns.intersection(df.columns)

	# Create a new list of data frames with only common columns
	new_data_frames = []
	for df in dfs:
		new_data_frames.append(df[common_columns])

	return new_data_frames

def get_users_in_nonempty_conversations(dfs):
    """
    Return the set of users who talked in non-empty conversations.
    - Checks the `user_list` column in the user-level dataframe
    - Filters out if the list of other users is empty (the user didn't have a true conversation)
    - Note that currently, the `user_list` is a list, cast as a string, so we use this format.

    @param dfs: list of dataframes (expecting User level!)
    """
    processed_dfs = []
    for df in dfs:
        processed_dfs.append(df[df['user_list'] != '[]'])
    return(processed_dfs)

def get_convs_with_min_value(dfs, col, min_value):
	"""
	Filters dataframes so that only conversations with a min value in a certain
	column remain.

	Example usage: Filtering dataframe to only those with a min. number of chats.

	@param: dfs: list of input dataframes
	@col: column being filtered on
	@min_value: minimum value desired for that column
	"""
	new_dfs = []
	for df in dfs:
		new_dfs.append(df[df[col]>min_value])
	return new_dfs

def get_pca_of_dataframes(dfs, n_components=None):
	"""
	Wrapper function for calling PCA on a list of dataframes.

	@param dfs: input DataFrames.
	@n_components: the number of components for the PCA

	Returns: the PCA object.
	"""
	# Stack and Normalize the Data
	stacked_data = pd.concat(get_numeric_cols(dfs), join = 'inner', ignore_index = True).transform(lambda x: (x - x.mean()) / x.std())
	# Remove Coluimns with NA
	stacked_data = stacked_data.dropna(axis=1)
	# Run PCA and return object
	pca = PCA(n_components=n_components)
	pca.fit_transform(stacked_data)
	return(pca)


def plot_pca_of_dataframes(*dfs, labels, title = "PCA Scatter Plot of DataFrames"):
	"""
	Plot data from multiple DataFrames on the same 2D PCA plot, coloring by labels.

	Parameters:
	*dfs (pd.DataFrame): Variable-length argument for input DataFrames.
	labels (list): List of labels corresponding to each DataFrame.

	Returns: the PCA object (so that the user can analyze what's going on with the data)
	"""
	# Add a "task_name" column to each DataFrame based on the corresponding label
	for i, df in enumerate(dfs):
		df['task_name'] = labels[i]
	
	# Concatenate the DataFrames into a single DataFrame
	stacked_data = pd.concat([df.assign(task_name=labels[i]) for i, df in enumerate(get_numeric_cols(dfs))], axis=0)

	# Normalize Columns Across All Tasks
	# This ensures that features with large numeric values don't skew the PCA
	# Simultaneously, normalizing *across* tasks ensures that 
	cols_to_normalize = [col for col in stacked_data.columns if col != 'task_name']
	normed_data = stacked_data[cols_to_normalize].transform(lambda x: (x - x.mean()) / x.std())

	# Drop columns with any NaN values
	normed_data = normed_data.dropna(axis=1)

	# Perform PCA for dimensionality reduction (2 components for 2D)
	pca = PCA(n_components=2)
	reduced_data = pca.fit_transform(normed_data)

	# Create a scatter plot and use the "task_name" column for coloring
	plt.figure(figsize=(10, 6))
	unique_labels = list(set(labels))
	colors = plt.cm.Dark2(np.arange(len(unique_labels)))

	for i, label in enumerate(unique_labels):
		label_data = reduced_data[stacked_data['task_name'] == label]
		plt.scatter(label_data[:, 0], label_data[:, 1], label=label, color=colors[i], alpha=0.6)

	plt.title(title)
	plt.xlabel("Principal Component 1")
	plt.ylabel("Principal Component 2")
	plt.legend(loc='best', title='Task')
	plt.grid(True)
	plt.show()

	return(pca)

def examine_top_n_components(pca, n):
    """
    Given a PCA object, examines the the top n features within the first two components
    (PC1 and PC2)

    @param pca: Scikit-learn PCA Ojbect
    @param n: Number of top features you desire to examine
    """
    feature_names = pca.feature_names_in_
    components = pca.components_
    # Create a DataFrame to display the data
    data = {
        "Feature": feature_names,
        "PC1": components[0],
        "PC2": components[1]
    }

    # Create the DataFrame
    df = pd.DataFrame(data)

    # Sort the DataFrame by the absolute values of PC1 and PC2
    df["PC1_abs"] = np.abs(df["PC1"])
    df["PC2_abs"] = np.abs(df["PC2"])

    # Select the top n features
    top_n_pc1 = df.sort_values(by=["PC1_abs"], ascending=False).iloc[:n, :]
    top_n_pc2 = df.sort_values(by=["PC2_abs"], ascending=False).iloc[:n, :]

    # Display the top n features for PC1
    print("Top 5 Features for PC1:")
    print(top_n_pc1)

    # Display the top n features for PC2
    print("\nTop 5 Features for PC2:")
    print(top_n_pc2)

    return(df)

def get_gaussian_mixture_clustering(data, use_aic = True):
	"""
	Usin Gaussian Mixture Models, get an optimal number of clusters for the data.

	@param data: The data in question;
	@param use_aic: use the optimal number of clusters from AIC (defaults to true);
		Otherwise, will use the number of clusters from BIC.
	"""
	# Define a range of cluster numbers to test
	n_components_range = range(1, 11)
	bic_scores = []
	aic_scores = []

	# Fit GMM models with different numbers of components
	for n_components in n_components_range:
	    gmm = GaussianMixture(n_components=n_components, random_state=42)
	    gmm.fit(data)
	    bic_scores.append(gmm.bic(data))
	    aic_scores.append(gmm.aic(data))

	# Plot BIC and AIC scores to find the optimal number of clusters
	plt.figure(figsize=(10, 5))
	plt.subplot(1, 2, 1)
	plt.plot(n_components_range, bic_scores, marker='o')
	plt.title('BIC Score')
	plt.xlabel('Number of Clusters')
	plt.ylabel('BIC Score')

	plt.subplot(1, 2, 2)
	plt.plot(n_components_range, aic_scores, marker='o')
	plt.title('AIC Score')
	plt.xlabel('Number of Clusters')
	plt.ylabel('AIC Score')

	plt.tight_layout()
	plt.show()

	# Choose the optimal number of clusters based on BIC or AIC (lower is better)
	optimal_n_components_bic = n_components_range[np.argmin(bic_scores)]
	optimal_n_components_aic = n_components_range[np.argmin(aic_scores)]

	# Fit the GMM with the optimal number of components
	if(use_aic):
		optimal_n = optimal_n_components_aic
	else:
		optimal_n = optimal_n_components_bic

	optimal_gmm = GaussianMixture(n_components=optimal_n, random_state=42)
	optimal_gmm.fit(data)

	# Assign cluster labels to the data
	cluster_labels = optimal_gmm.predict(data)

	return(cluster_labels)


def generate_interactive_feature_plot(pca_df, title = "PCA Scatter Plot"):
	"""
	Generate a plot of the PCA's, in which a hovertool allows the user to look at the
	different features within each cluster

	@param pca_df: dataframe that results from running PCA
	"""
	source = ColumnDataSource(pca_df)

	# Define a color palette and map cluster values to colors
	palette = Viridis256  # You can choose any other palette from Bokeh
	color_mapper = linear_cmap(field_name='cluster', palette=palette, low=min(pca_df['cluster']), high=max(pca_df['cluster']))

	# Create a Bokeh figure
	output_file("pca_scatter_plot.html")  # Output file name
	p = figure(title=title)

	# Scatter plot with colored clusters
	scatter = p.circle('PC1', 'PC2', source=source, size=10, color=color_mapper, legend_field='cluster')

	# Add tooltips using the feature names
	hover = HoverTool()
	hover.tooltips = [("Feature", "@index"), ("Cluster", "@cluster")]
	p.add_tools(hover)

	p.legend.title = 'Cluster'
	p.legend.label_text_font_size = '10pt'
	show(p)


def visualize_feature_clusters(dfs, use_aic = True, title = "PCA Scatter Plot"):
	"""
	A function that takes in the list of dataframes and visualizes how the different features
	clusters together.

	@param dfs: list of dataframes
	@param use_aic: a boolean to indicate whether we cluster with the optimal number of clusters
	using AIC or BIC
	"""
	all_data = pd.concat(dfs, join = 'inner', ignore_index = True)

	# Normalize (Across all tasks)
	columns_to_normalize = [col for col in all_data.columns if col != 'task_name']
	normalized_df = all_data.copy()
	normalized_df[columns_to_normalize] = normalized_df[columns_to_normalize].transform(lambda x: (x - x.mean()) / x.std())

	# Drop NA columns
	nan_cols, nan_rows = check_na_rows_cols(normalized_df)
	normalized_df = normalized_df.drop(nan_cols, axis = 1)

	normalized_data_transposed = normalized_df.drop(["task_name"], axis=1).T

	# Get Clusters (using GMM) for the data, on the full-dimensional dataset.
	cluster_labels = get_gaussian_mixture_clustering(normalized_data_transposed, use_aic)

	# Perform PCA for dimensionality reduction (2 components for 2D)
	pca = PCA(n_components=2)
	pca_data_transposed = pca.fit_transform(normalized_data_transposed)

	# Create a new DataFrame with the PC scores and row names
	pca_df = pd.DataFrame(data=pca_data_transposed, columns=["PC1", "PC2"], index=normalized_data_transposed.index)
	pca_df["cluster"] = cluster_labels

	generate_interactive_feature_plot(pca_df, title)

