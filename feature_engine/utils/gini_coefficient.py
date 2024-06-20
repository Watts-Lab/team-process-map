import numpy as np
import csv
import pandas as pd
 
def gini_coefficient(x):
    """
    Calculates the Gini coefficient for an array of values, which is a measure of statistical dispersion.

    Source code: https://stackoverflow.com/questions/39512260/calculating-gini-coefficient-in-python-numpy

    :param x: List or array of values to calculate the Gini coefficient for.
    :type x: list or np.ndarray
    :return: Gini coefficient value.
    :rtype: float
    """
    diffsum = 0
    for i, xi in enumerate(x[:-1], 1):
        diffsum += np.sum(np.abs(xi - x[i:]))
    if ((len(x)**2 * np.mean(x)) == 0):
        return np.nan
    return diffsum / (len(x)**2 * np.mean(x))

def get_gini(input_data, on_column):
	"""
    Calculates the Gini coefficient for a specified numeric column within grouped conversation data.

    :param input_data: A DataFrame of conversations, where each row represents one chat.
    :type input_data: pd.DataFrame
    :param on_column: The name of the numeric column on which the Gini coefficient is to be calculated.
    :type on_column: str
    :return: A DataFrame with Gini coefficients for each conversation.
    :rtype: pd.DataFrame
    """

	gini_calculated = input_data.groupby(["conversation_num"]).apply(lambda df : gini_coefficient(np.asarray(df[on_column]))).reset_index().rename(columns={0: "gini_coefficient_" + on_column})
	return(gini_calculated)