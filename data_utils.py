import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer

def get_null_stats(df):
    null_count = df.isnull().sum()
    col_null_counts = {col: null_cnt for col, null_cnt in null_count.iteritems() if null_cnt > 0}
    if len(col_null_counts) == 0:
        print(f"There are no columns with null values")
    else:
        print(col_null_counts)

def get_outliers_count(df, col_name):
    """
    Calculate the count of outliers in a given column of a DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
        col_name (str): The name of the column to calculate outliers for.

    Returns:
        int: The count of outliers in the specified column.
    """
    return df[col_name].loc[df[col_name] > df[col_name].mean() + 3 * df[col_name].std()].count()


def process_outliers_iqr(df, col_name, remove_outliers=True):
    """
    Process outliers in a DataFrame column using the Interquartile Range (IQR) method.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
        col_name (str): The name of the column to process outliers for.
        remove_outliers (bool, optional): Flag to indicate whether to remove outliers. Defaults to True.

    Returns:
        pandas.DataFrame, pandas.DataFrame: Processed DataFrame after outlier removal and a DataFrame with outlier analysis results.
    """
    Q1 = df[col_name].quantile(0.25)
    Q3 = df[col_name].quantile(0.75)
    IQR = Q3 - Q1    
    min_val = Q1 - 1.5 * IQR
    max_val = Q3 + 1.5 * IQR    
    outlier_count = df[(df[col_name] < min_val) | (df[col_name] > max_val)].shape[0]
    if remove_outliers:
        df = df[(df[col_name] >= min_val) & (df[col_name] <= max_val)]
    # Create a DataFrame for the results
    result = pd.DataFrame({
        'col_name': [col_name],
        'Q1': [Q1],
        'Q3': [Q3],
        'IQR': [IQR],
        'min_val': [min_val],
        'max_val': [max_val],
        'outlier_count': [outlier_count],
        'outlier_pct': [outlier_count / df.shape[0]]
    })    
    return df, result

def power_transform(df, col_name, skew_threshold=0.5):    
    transformed = False
    skew = df[col_name].skew()
    print(f"{col_name} has skewness of {skew}")
    power_transformer = PowerTransformer(method='yeo-johnson', standardize=True)    
    if abs(skew) > skew_threshold:
        transformed = True
        print("Will apply power transform.")
        col_transformed = power_transformer.fit_transform(df[[col_name]])
        df.loc[:, col_name] = col_transformed
    return df, transformed