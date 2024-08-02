import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

def plot_feature_distribution(train_df, test_df, cols_float, fig_size):
    # plot the distribution of numerical features . Also check if train and  test data have roughly 
    # the same distribution for numerical features
    n_features = len(cols_float)
    n_rows = (n_features + 1) // 2  # Integer division for ceiling    
    fig, axes = plt.subplots(nrows=n_rows, ncols=2, figsize=fig_size, dpi=100)    
    fig.suptitle('Distribution of Features in Train and Test Sets', fontsize=12)
    plt.subplots_adjust(hspace=0.3)
    # Loop through features and create subplots
    for i, col_name in enumerate(cols_float):
        row = i // 2
        col = i % 2        
        sns.histplot(x=train_df[col_name], label="Train", kde=True, fill=True, color="orange", ax=axes[row, col])
        sns.histplot(x=test_df[col_name], label="Test", kde=True, fill=True, color="teal", ax=axes[row, col])        
        axes[row, col].legend()
        axes[row, col].set_ylabel("count")
        axes[row, col].set_xlabel(col_name)                
    # Remove extra subplots if the number of features is odd
    if n_features % 2 == 0:
        fig.delaxes(axes[-1, -1])  # Delete the last subplot if there's an empty one
    fig.tight_layout()
    plt.show()   

def plot_box_plots(train_df, test_df, cols_float, fig_size):
    n_rows = len(cols_float)
    n_cols = 2
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=fig_size, dpi=100)    
    plt.subplots_adjust(hspace=0.3)
    for row in range(n_rows):
        col = 0
        col_name = cols_float[row]
        sns.boxplot(x=train_df[col_name], orient='v', color='skyblue', ax=axes[row, col])
        axes[row, 0].set_ylabel("")
        axes[row, 0].set_xlabel(col_name)
        axes[row, 0].set_title(f'Train', fontsize=12)
        col=1
        sns.boxplot(x=test_df[col_name], orient='v', color='skyblue', ax=axes[row, col])
        axes[row, 1].set_ylabel("")
        axes[row, 1].set_xlabel(col_name)
        axes[row, 1].set_title(f'Test', fontsize=12)
    fig.tight_layout()
    plt.show()       

def plot_feature_target_corr(df, feature_cols, target_col):
    fig, ax = plt.subplots(figsize=(14, 6))
    df = df[feature_cols + [target_col]]
    corr = df.corr()
    target_feature_interaction = corr[target_col].sort_values(ascending=False)
    labels = target_feature_interaction.index.to_list()
    labels.remove(target_col)
    values = target_feature_interaction.values.tolist()
    values.pop(0)
    ax.set_title("Feature target correlation")
    ax.set_xlabel(f"{target_col} correlation")
    ax.set_ylabel("Features")
    ax = sns.barplot(x=values, y=labels, ax=ax)    

# df_feature_imp is a dataframe with two columns f_name and f_imp
def plot_feature_importance(df_feature_imp, fig_size=(18, 6)):
    # Set the figure size
    plt.figure(figsize=(18, 6))
    # Create the bar plot
    sns.barplot(x="f_name", y="f_imp", data=df_feature_imp)
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=90)
    # Add feature importance values on top of each bar (adjusted positioning)
    for bar in plt.gca().containers[0]:
      plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2.0, round(bar.get_height(), 3), 
           ha='center', va='center', rotation=90)  # Adjust vertical offset
    # Customize plot title and labels (optional)
    plt.title("Feature Importance")
    plt.xlabel("Feature Name")
    plt.ylabel("Feature Importance")
    # Show the plot
    plt.show()

def reduce_df_memory(df: pd.DataFrame):
    "This method reduces memory for numeric columns in the dataframe"
    
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64', "uint16", "uint32", "uint64"]
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtypes
        
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()

            if "int" in str(col_type):
                if c_min >= np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min >= np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min >= np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min >= np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min >= np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                if c_min >= np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)  

    end_mem = df.memory_usage().sum() / 1024**2
    print(f"Start - end memory:- {start_mem:5.2f} - {end_mem:5.2f} Mb")
    return df    