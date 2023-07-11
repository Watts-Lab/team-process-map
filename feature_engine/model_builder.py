# 3rd Party Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import shap
# TODO Imports needed for model optimization - will be useful in a PR dedicated for optimization
# import optuna
# from functools import partial

# Note -- running this class requires Python 3.9 or earlier (not compatible with more recent Python)

class ModelBuilder():
    def __init__(
            self, 
            config_path: str="config.json", 
            output_dir: str="output/", 
            task_map_path: str='utils/task_map.csv', 
            dataset_names: list=["csop"], 
            test_dataset_names: list=None,
            standardize_within = False
        ) -> None:
        """Initializes the various objects and variables used throughout the `ModelBuilder` class.

        Args:
            config_path (str, optional): Path to the `config.json` file which has details like the dataset names, irrelevant columns for each dataset etc.. Defaults to `"config.json"`.
            output_dir (str, optional): Path to the outputs directory. This is where the datasets to be used by the model are stored. Defaults to `"output/"`.
            task_map_path (str, optional): Path to the `.csv` file that contains task related features. Defaults to `'utils/task_map.csv'`.
            dataset_names (list, optional): A `list` of `strings` that describe the dataset names as present in the `config.json` file. Defaults to `["csop"]`.
            test_dataset_names (list, optional): A list of `strings` that allow users to designate one of the datasets as the test dataset. 
                                                 Ideally this value would be set to `["csopII"]` when `dataset_names` is set to `["csop"]`. Otherwise it is wise to keep it None.
                                                 Defaults to `None`.
            standardize_within (bool, optional): A boolean that determines whether features are standardized *within* tasks /individual datasets (if True), or *across* tasks (if False). Defaults to False.

        
        Returns:
            (None)
        """
        # Read in all the necessary file paths and column names from the config file
        with open(config_path, "rb") as json_file:
            self.config = json.load(json_file)
            self.dataset_names = dataset_names
            self.conv = None
            self.convs_complete = [] # list version of conv_complete 

            self.test_dataset_names = test_dataset_names
            self.test_conv = None
            self.test_convs_complete = [] # list version of conv_complete 

            self.output_dir = output_dir
            self.task_maps = pd.read_csv(task_map_path)
            self.task_maps = self.task_maps[self.task_maps['task'].isin(self.config['task_names'])]
            self.standardize_within = standardize_within
        # If the user has specified a test set(s), then we need to build it seperately
        if self.test_dataset_names != None: 
            self.create_datasets(is_test_datasets=True)
        # Use this function to create the dataset for model training
        self.create_datasets(is_test_datasets=False)
        
        # Defining placeholders for the features, targets and the baseline model
        self.X, self.y = None, None
        self.baseline_model = None
        # TODO - Will work on model optimization in a subsequent PR
        # self.optimized_model = None

    def create_datasets(self, is_test_datasets: bool=False) -> None:
        """This function allows the user to simply specify the names of the datasets needed in training and testing. Using just the names, 
           it combines the relevant datasets together into relevant dataframes for model training/testing.

        Args:
            is_test_datasets (bool, optional): If the user specifies a test dataset, then this function is invoked again to combine the datasets for testing, 
                                               but this time the results are stored in the test dataframes (`self.test_conv`) rather than the train ones (`self.conv`). 
                                               Defaults to False.

        Raises:
            TypeError: If the dataset names are not in the format consistent with the `config.json` file, this function raises an exception
        
        Retrunns:
            (None)
        """
        # If we are creating the testing dataframe, then take in the dataset names from the testing list, else take the names from the training list of dataset names.
        if is_test_datasets: dataset_names = self.test_dataset_names
        else: dataset_names = self.dataset_names
        
        convs_complete = []
        conv = None
        conv_complete = None

        # List to store whether timestamp is present
        has_timestamp = []

        # Exception to handle the case when the dataset names are not encapsulated in a list
        if(not isinstance(dataset_names, list)):
            raise TypeError("Please provide the dataset names as a list!")

        # If we have only one dataset then there is not need to combine datasets. Just remove the redundant columns from the dataset needed, and we have the required dataset.
        if(len(dataset_names)==1): 
            try:
                conv_complete = pd.read_csv(self.output_dir + self.config[dataset_names[0]]["filename"])
            # Handling the case when the dataset names is not found in the config files
            except KeyError:
                print("Are you sure that your dataset name is correct?")
            # Dropping redundant columns as specified in the config file    
            conv = conv_complete.drop(self.config[dataset_names[0]]["cols_to_ignore"], axis=1).dropna()
            # Standardizing Features
            conv = pd.DataFrame(StandardScaler().fit_transform(conv),columns = conv.columns)
            # We store the dataset name to make it easy to join task related features later on in the pipeline.
            conv['dataset_name'] = dataset_names[0]

        # This block is used to combine multiple datasets together if the user specified more than one dataset name.
        else: 
            # For each dataset name specified:
            for dataset_name in dataset_names:
                # Try reading in the dataset
                try:
                    full_dataset = pd.read_csv(self.output_dir + self.config[dataset_name]["filename"]).dropna()
                    convs_complete.append(full_dataset)
                # Handling the case when the dataset names is not found in the config files
                except KeyError:
                    print("Are you sure that your dataset name is correct?")
                
                # Remove redundant columns from the current dataset
                df_extra_columns_dropped = full_dataset.drop(self.config[dataset_name]["cols_to_ignore"], axis=1).dropna()
                
                # check if timestamp is present
                has_timestamp.append("timestamp" in self.config[dataset_name]["cols_to_ignore"])

                #Standard Features WIHIN each task
                if self.standardize_within:
                    df_extra_columns_dropped = pd.DataFrame(StandardScaler().fit_transform(df_extra_columns_dropped),columns = df_extra_columns_dropped.columns)

                # Add a column with the dataset name --- This is helpful for adding in task related featured in the pipeline later on.
                df_extra_columns_dropped = df_extra_columns_dropped.assign(dataset_name = dataset_name)
  
                # merge with self.conv
                if conv is None: conv = df_extra_columns_dropped
                else: conv = pd.concat([conv, df_extra_columns_dropped])
        
        #Standard Features ACROSS each task
        if not self.standardize_within:
            # standardize only numeric cols
            numeric = conv[conv.select_dtypes(include='number').columns]
            scaled_numeric = pd.DataFrame(StandardScaler().fit_transform(numeric), columns=numeric.columns)
            conv[numeric.columns] = scaled_numeric.values

        # Remove timestamp if not present in all datasets being concatenated
        if(not all(has_timestamp)):
            conv = conv.drop(columns=[col for col in conv.columns if 'time_diff' in col.lower()])

        # Handle / store test dataset
        if is_test_datasets: 
            # This is the dataset that is used for model testing (if the user specified a test set that is. if not, then the training set is broken into train-val-test sets)
            self.test_conv = conv
            # When we have multiple datasets, we have self.convs_complete
            self.test_conv_complete = conv_complete
            # When we have a single dataset, we have only self.conv_complete
            self.test_convs_complete = convs_complete
        
        # Handle / store training dataset
        else:
            # This is the dataset that is used for model training
            self.conv = conv
            # When we have multiple datasets, we have self.convs_complete
            self.convs_complete = convs_complete
            # When we have a single dataset, we have only self.conv_complete
            self.conv_complete = conv_complete
        
        # Calling this function to add in task related features to the combined datasets created above.
        self.integrate_task_level_features(is_test_datasets=is_test_datasets)

    def integrate_task_level_features(self, is_test_datasets: bool=False) -> None:
        """This function takes in the datasets created by `create_datasets()` function and adds in the task related features to it.

        Args:
            is_test_datasets (bool, optional): A boolean flag indicating whether the operation needs to be performed on the training or the testing set. Defaults to False.
        """
        # For both train or test sets, we join the datasets created by `create_datasets()` function with the task related features on the task name that was added in by the 
        # `create_datasets()` function. To ensure a successful join, the config provides a map between the dataset names used in the conversation level datasets and the task 
        # features dataset.
        if is_test_datasets: 
            self.test_conv['task_name'] = self.test_conv['dataset_name'].map(self.config['task_mapping_keys'])
            self.test_conv = pd.merge(left=self.test_conv, right=self.task_maps, left_on=['task_name'], right_on=['task'], how='left')
            self.test_conv.drop(['dataset_name', 'task_name', 'task'], axis=1, inplace=True)
        else:
            self.conv['task_name'] = self.conv['dataset_name'].map(self.config['task_mapping_keys'])
            self.conv = pd.merge(left=self.conv, right=self.task_maps, left_on=['task_name'], right_on=['task'], how='left')
            self.conv.drop(['dataset_name', 'task_name', 'task'], axis=1, inplace=True)

    def set_target(self, target: list, is_test: bool) -> None:
        """This function is used to set the targets for `self.conv` and `self.test_conv`. We allow the user to choose the column 
           (from the input datasets) which they want to treat as the target. This function facilitates the addition of the targets 
           to the dataframes that are going to be used by the ML models later on.

        Args:
            target (list): A list of strings that are the column names to be used as targets in the final models
            is_test (bool): A boolean flag indicating whether we are dealing with the train or the test set.
        """
        # Setting the local variables to test or train sets depending on the boolean flag `is_test`
        if(not is_test):
            conversation_clean = self.conv
            conversation_complete = self.conv_complete
            conversation_complete_list = self.convs_complete
        else:
            conversation_clean = self.test_conv
            conversation_complete = self.test_conv_complete
            conversation_complete_list = self.test_convs_complete

        # In case we have one dataset only
        if(len(self.dataset_names)==1):
            # In case user provided target as list
            if(isinstance(target, list)): target = target[0]
            # The raw target is stored in `target_raw`, and the stadardized target is stored in `target_std`.
            conversation_clean["target_raw"] = conversation_complete[target]
            conversation_clean["target_std"] = (conversation_clean["target_raw"] - conversation_clean["target_raw"].mean()) / conversation_clean["target_raw"].std()
        # In case we are combining multiple datasets and targets
        else: 
            target_raw_list, target_std_list = [], []
            # For each target, fetch its raw values from the respective dataset in which the target is located, and calculate its standardized version
            for target_idx, target_name in enumerate(target):
                target_raw_list.extend(conversation_complete_list[target_idx][target_name].to_list())
                scaler = StandardScaler()
                target_std_list.extend(scaler.fit_transform(conversation_complete_list[target_idx][target_name].to_numpy().reshape(-1, 1)).flatten())
            # Add in the combined raw and standardized targets
            conversation_clean["target_raw"] = target_raw_list
            conversation_clean["target_std"] = target_std_list

        # Set everything in global variables of the class
        if(not is_test):
            self.conv = conversation_clean
            self.conv_complete = conversation_complete
            self.convs_complete = conversation_complete_list
        else:
            self.test_conv = conversation_clean
            self.test_conv_complete = conversation_complete
            self.test_convs_complete = conversation_complete_list
            
    def select_target(self, target: list) -> None:
        """This function calls the `set_target()` above for the train set.

        Args:
            target (list): A list of strings that are the column names to be used as targets in the final models
        """
        self.set_target(target, is_test = False)

    def select_test_target(self, target: list):
        """This function calls the `set_target()` above for the test set.

        Args:
            target (list): A list of strings that are the column names to be used as targets in the final models
        """
        self.set_target(target, is_test = True)

    def viz_target(self) -> None:
        """This function creates 4 graphs to visualize the chosen target(s). 
            1. The distribution of the chosen target 
            2. The distribution of the standardized target 
            3. A heatmap of the correlation between all features and raw target
            4. A heatmap of the correlation between all features and standardized target
        """
        fig, ax = plt.subplots(1, 2, figsize=(15, 4))
        ax[0].hist(self.conv["target_raw"])
        ax[0].set(xlabel="Chosen Target (Raw)", ylabel="Frequency", title="Distribution of Chosen Target (Raw)")
        ax[0].grid()

        ax[1].hist(self.conv["target_std"])
        ax[1].set(xlabel="Chosen Target (Std)", ylabel="Frequency", title="Distribution of Chosen Target (Std)")
        ax[1].grid()
        plt.show()

        fig, ax = plt.subplots(1, 2, figsize=(20, 7))
        sns.heatmap(self.conv.drop(["target_std"], axis=1).corr(), cmap="RdBu", ax=ax[0])
        ax[0].set(title="Feature Correlations with Raw Target")
        ax[0].axis("off")

        sns.heatmap(self.conv.drop(["target_raw"], axis=1).corr(), cmap="RdBu", ax=ax[1])
        ax[1].set(title="Feature Correlations with Standardized Target")
        ax[1].axis("off")

    
    def define_model(self, model_type: str="xgb") -> None:
        """Instantiates a model based on the keyword specified by the user.

        Args:
            model_type (str, optional): A string keyword that allows us to define a model object of the type desired by the user. 
                                        Current model types supported are: `['xgb', 'lasso', 'linear', 'rf']`
                                        Defaults to "xgb".
        """
        self.model_type = model_type
        if model_type == 'xgb':
            self.baseline_model = XGBRegressor(random_state=42)
        elif model_type == 'lasso':
            self.baseline_model = LassoCV(alphas = [0.01, 0.05, .1, 0.25, 0.5, 0.75, 1])
        elif model_type == 'linear':
            self.baseline_model = LinearRegression()
        elif model_type == 'rf':
            self.baseline_model = RandomForestRegressor(random_state=42)

    def define_dataset_for_model(self, is_test: bool=False, fit_on_raw: bool=False) -> list:
        """This function stores the features and targets defined in the various global variables into the standard notations - `X`, and `y`. 
           It handles the creation of X, and y matrices for when the user specifies a specific test dataset and otherwise. 
           It also allows users to regress on standardized or raw versions of the target.

        Args:
            is_test (bool, optional): Boolean flag that denotes whether we are working with a user defined test set or not. Defaults to False.
            fit_on_raw (bool, optional): Boolean flag that denotes whether we need to regress on standardized or raw versions of the target. Defaults to False.

        Returns:
            list: Return the list of features (`X`) and targets (`y`) formated as - `[X, y]`
        """
        if is_test:
            # Make sure both self and test have timestamp; otherwise, drop time-related columns
            if (len([col for col in self.conv.columns if 'time_diff' in col.lower()]) == 0): # train set has no timestamp
                if(len([col for col in self.test_conv.columns if 'time_diff' in col.lower()]) > 0): # but test set has timestamp
                    self.test_conv = self.test_conv.drop(columns=[col for col in self.test_conv.columns if 'time_diff' in col.lower()])

            if fit_on_raw: X, y = self.test_conv.drop(["target_raw", "target_std"], axis=1), self.test_conv["target_raw"]
            else: X, y = self.test_conv.drop(["target_raw", "target_std"], axis=1), self.test_conv["target_std"]

        else:
            if fit_on_raw: X, y = self.conv.drop(["target_raw", "target_std"], axis=1), self.conv["target_raw"]
            else: X, y = self.conv.drop(["target_raw", "target_std"], axis=1), self.conv["target_std"]
        # TODO - This might be redundant now that the dataset names are replaced by task level features
        # Get one hot encodings of any object column
        X = pd.get_dummies(X)

        return X, y

    def evaluate_model(self, model, val_size: float=0.1, test_size: float=0.1) -> None:
        """This is a driver function that calls a bunch of different functions for the following tasks:
           - Train-Val-Test splits
           - Model Training
           - Printing Model Metrics on the 3 sets
           - Plotting Feature Importances and SHAP plots

        Args:
            model (sklearn/xgboost model): This is the model object defined by `define_model()` function
            val_size (float, optional): This controls the validation data size. Defaults to 0.1.
            test_size (float, optional): This control the test data size (used when there was no explicit dataset designated for testing). Defaults to 0.1.
        """
        print('Checking Holdout Sets', end='...')
        if self.test_dataset_names == None:
            print('Creating Holdout Sets...')
            self.create_holdout_sets(val_size=val_size, test_size=test_size)
        else:
            self.create_holdout_sets(val_size=val_size)
        print('Done')
        print('Training Model', end='...')
        model = model.fit(self.X_train, self.y_train)
        print('Done')
        self.summarize_model_metrics(model)
        self.model_diagnostics(model)

    def create_holdout_sets(self, val_size: float=0.1, test_size: float=None) -> None:
        """This function splits the data into train-val-test sets if no test set is designated, and splits into train-val only if we already have a designated test set.

        Args:
            val_size (float, optional): This controls the percentage of rows out of the train set that need to be held out for validation. Defaults to 0.1.
            test_size (float, optional): This controls the percentage of rows out of the entire dataset that need to be held out for testing. Defaults to None.
        """
        # Calls the `define_dataset_for_model()` function to get the `X`s and the `y`s
        self.X, self.y = self.define_dataset_for_model()
        # If we need to split out a test set
        if test_size:
            # Spit train-val and test set in a `(1-test_size)-test_size` split
            X_train_val, X_test, y_train_val, y_test = train_test_split(self.X, self.y, random_state=42, test_size=test_size)
            # Spit train and val set in a `(1-val_size)-val_size` split (here split sizes refers to the percentages after the split done above)
            X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, random_state=42, test_size=val_size)
            # Set the train val and test sets in global variables to be used by all classes
            self.X_train, self.y_train = X_train, y_train
            self.X_val, self.y_val = X_val, y_val
            self.X_test, self.y_test = X_test, y_test
        # If we have a designated test set
        else:
            # Spit train and val set in a `(1-val_size)-val_size` split (here split sizes refers to the percentages after the split done above)
            X_train, X_val, y_train, y_val = train_test_split(self.X, self.y, random_state=42, test_size=val_size)
            # Set the train val and test sets in global variables to be used by all classes
            self.X_train, self.y_train = X_train, y_train
            self.X_val, self.y_val = X_val, y_val
            # Set the test set by preprocessing the `test_conv` global variable in the `define_dataset_for_model()` function.
            self.X_test, self.y_test = self.define_dataset_for_model(is_test=True)

    def summarize_model_metrics(self, model) -> None:
        """Prints out model metrics for each of the datasets - train, val and test

        Args:
            model (sklearn/xgboost model): The fitted model.
        """
        train_metrics = self.calculate_model_metrics(model=model, dataset=(self.X_train, self.y_train))
        val_metrics = self.calculate_model_metrics(model=model, dataset=(self.X_val, self.y_val))
        test_metrics = self.calculate_model_metrics(model=model, dataset=(self.X_test, self.y_test))
        print("MODEL METRICS")
        print('Train Set:', end='\t')
        print('R2: {}\tMAE: {}\tMSE: {}\tRMSE: {}'.format(train_metrics['r2'], train_metrics['mae'], train_metrics['mse'], train_metrics['rmse']))
        
        print('Validation Set:', end='\t')
        print('R2: {}\tMAE: {}\tMSE: {}\tRMSE: {}'.format(val_metrics['r2'], val_metrics['mae'], val_metrics['mse'], val_metrics['rmse']))

        print('Test Set:', end='\t')
        print('R2: {}\tMAE: {}\tMSE: {}\tRMSE: {}'.format(test_metrics['r2'], test_metrics['mae'], test_metrics['mse'], test_metrics['rmse']))

    def calculate_model_metrics(self, model, dataset: list) -> dict:
        """Returns a dictionary with the model metrics to be printed out by the summarize_model_metrics() function.

        Args:
            model (sklearn/xgboost model): Fitted model
            dataset (list): X, y matrices

        Returns:
            dict: A dictionary with the metric values for the given dataset
        """
        X, y = dataset
        r2 = r2_score(y_true=y, y_pred=model.predict(X)).round(4)
        mae = mean_absolute_error(y_true=y, y_pred=model.predict(X)).round(4)
        mse = mean_squared_error(y_true=y, y_pred=model.predict(X)).round(4)
        rmse = np.sqrt(mse).round(4)
        return {'r2': r2, 'mae': mae, 'mse': mse, 'rmse': rmse}

    def model_diagnostics(self, model) -> None:
        """Plots the feature importances and the SHAP summary scores for the top features for the fitted model.

        Args:
            model (sklearn/xgboost model): Fitted model
        """
        # Model diagnostics for tree based models
        if self.model_type in ['xgb', 'rf']:
            explainer = shap.TreeExplainer(model)
        # Model diagnostics for linear models
        elif self.model_type in ['lasso', 'linear']:
            if self.model_type == 'lasso':
                self.print_lasso_coefs(model)
                self.plot_lasso_residuals(model)
            explainer = shap.LinearExplainer(model, self.X_val)
            
        shap_values = explainer.shap_values(self.X_val)
        shap.summary_plot(shap_values, self.X_val, feature_names=self.X.columns, plot_type="bar")
        shap.summary_plot(shap_values, self.X_val, feature_names=self.X.columns)

    def print_lasso_coefs(self, model):

        coefficients = model.coef_
        
        # Get coefficients that are larger than minimal thresh = 0.001
        # This drops the worst coefficients in the graph and makes the image more readable
        MIN_THRESH = 0.01
        if(len(coefficients[abs(coefficients) > MIN_THRESH])>0):
            nonzero_coefficients,nonzero_indices = self.get_nonzero_lasso_coefs(coefficients, MIN_THRESH)
        else:
            MIN_THRESH = 0
            nonzero_coefficients,nonzero_indices = self.get_nonzero_lasso_coefs(coefficients, MIN_THRESH)

        nonzero_names = self.X.columns[nonzero_indices] 
        sorted_indices = np.argsort(np.abs(nonzero_coefficients))[::-1]
        sorted_coefficients = nonzero_coefficients[sorted_indices]
        sorted_names = nonzero_names[sorted_indices]

        fig, ax = plt.subplots(figsize=(20, 7))
        ax.barh(range(len(sorted_coefficients)), sorted_coefficients, color='steelblue')
        ax.invert_yaxis()

        coeff_range = np.abs(np.max(sorted_coefficients) - np.min(sorted_coefficients))
        offset = coeff_range * 0.03

        for i, (coefficient, name) in enumerate(zip(sorted_coefficients, sorted_names)):
            text_x = coefficient - offset if coefficient < 0 else coefficient + offset
            ax.text(text_x, i, f'{coefficient:.2f}', ha='left' if coefficient < 0 else 'right', va='center')

        ax.set_yticks(range(len(sorted_coefficients)))
        ax.set_yticklabels(sorted_names)
        ax.set_xlabel('Coefficient Value')
        ax.set_ylabel('Coefficient Name')
        ax.set_title('Lasso Coefficients with CV-fitted alpha = ' + str(model.alpha_))

        plt.show()

    def get_nonzero_lasso_coefs(self, coefficients, MIN_THRESH):
        nonzero_coefficients = coefficients[abs(coefficients) > MIN_THRESH]
        nonzero_indices = np.where(abs(coefficients) > MIN_THRESH)[0]
        return(nonzero_coefficients,nonzero_indices)

    def plot_lasso_residuals(self, model):
   
        predicted_values = model.predict(self.X)
        residuals = self.y - predicted_values
        
        plt.figure(figsize=(8, 6))
        plt.scatter(predicted_values, residuals)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        plt.show()

    # TODO - Will deal with optimization workstream in a subsequent PR
    # def optimize_model(self, n_trials=15):
    #     optimization_function = partial(self.optimize)
    #     study = optuna.create_study(direction="minimize")
    #     # TODO - curerntly optimization runs with a fixed number of trials. Can we create a more intelligent stopping criterion?
    #     # Also TODO -- seems like for the most part, optimized models actually *underperform* baseline models
    #     # possible we need more iterations? 
    #     study.optimize(optimization_function, n_trials=n_trials)
    #     self.optimized_model = XGBRegressor(**study.best_params, random_state=42, objective="reg:squarederror")
    
    # def optimize(self, trial):
    #     """
    #     Function used to produce optimized (Bayesian Optimization) Random Forests regressor
        
    #     INPUTS:
    #         :trial: One Optimization trial.
    #         :X_train (np.ndarray): Training data
    #         :y_train (np.ndarray): Training labels
    #         :X_test (np.ndarray): Testing data
    #         :y_test (np.ndarray): Testing labels
    #     """
    #     # Define Parameter Space
    #     criterion = trial.suggest_categorical("criterion", ["absolute_error", "squared_error", "poisson"])
    #     n_estimators = trial.suggest_int("n_estimators", 100, 1500)
    #     max_depth = trial.suggest_int("max_depth", 3, 15)
    #     max_features = trial.suggest_uniform("max_features", 0.01, 1.0)
    #     max_samples = trial.suggest_uniform("max_samples", 0.01, 1.0)
    #     learning_rate = trial.suggest_uniform("learning_rate", 0.001, 0.01)
    #     if self.model_type == 'xgb': gamma = trial.suggest_uniform("gamma", 0.001, 0.02)

    #     # Define Model
    #     if self.model_type == 'xgb':
    #         model = XGBRegressor(
    #             n_estimators=n_estimators,
    #             max_depth=max_depth,
    #             max_features=max_features,
    #             criterion=criterion,
    #             max_samples=max_samples,
    #             learning_rate=learning_rate,
    #             gamma=gamma,
    #             random_state=42, 
    #             objective="reg:squarederror"
    #         )
    #     elif self.model_type == 'rf': # TODO -- despite 'rf', this fails with error
    #         model = RandomForestRegressor(
    #             n_estimators=n_estimators,
    #             max_depth=max_depth,
    #             max_features=max_features,
    #             criterion=criterion,
    #             max_samples=max_samples,
    #             learning_rate=learning_rate,
    #             random_state=42
    #         )

    #     # TODO -- throw an error if user provides a model type (e.g., linear) that cannot be optimized
        
    #     mse = -1*cross_val_score(estimator=model, X=self.X, y=self.y, scoring=make_scorer(mean_squared_error, greater_is_better=False, cv=self.folds)).mean().round(4)

    #     # Return MSE
    #     return mse