from sklearn import model_selection
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from skmultilearn.model_selection import IterativeStratification
from skmultilearn.model_selection.measures import get_combination_wise_output_matrix
from collections import Counter

def strat_group_kfold_dataframe(df, target_col_name, group_col_name, random_state, num_folds=5):
    # we create a new column called kfold and fill it with -1
    df["kfold"] = -1
    # randomize of shuffle the rows of dataframe before splitting is done
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    # get the target data
    y = df[target_col_name].values    
    groups = df[group_col_name].values
    # stratify data using anchor as group and score as target
    skf = model_selection.StratifiedGroupKFold(n_splits=num_folds, shuffle=True, random_state=random_state)
    for fold, (train_index, val_index) in enumerate(skf.split(X=df, y=y, groups=groups)):
        df.loc[val_index, "kfold"] = fold        
    return df            

def strat_kfold_dataframe(df, target_col_name, random_state, num_folds=5):
    # we create a new column called kfold and fill it with -1
    df["kfold"] = -1
    # randomize of shuffle the rows of dataframe before splitting is done
    df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    y = df[target_col_name].values
    skf = model_selection.StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_state)
    # stratification is done on the basis of y labels, a placeholder for X is sufficient
    for fold, (train_idx, val_idx) in enumerate(skf.split(X=df, y=y)):
        df.loc[val_idx, "kfold"] = fold
    return df     

# This method uses the iterstrat library for multilabel stratification
def iterstrat_multilabel_stratified_kfold_cv_split(df_train, label_cols, num_folds, random_state):
    mskf = MultilabelStratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_state)    
    df_targets = df_train[label_cols]
    for fold, (train_index, val_index) in enumerate(mskf.split(df_train["id"], df_targets)):        
        df_train.loc[val_index, "kfold"] = fold
    return df_train

# This method uses the skmultilearn library for multilabel stratification
def skml_multilabel_stratified_kfold_cv_split(df_train, label_cols, num_folds):
    mskf = IterativeStratification(n_splits=num_folds, order=1)
    X = df_train["id"]
    y = df_train[label_cols]
    for fold, (train_index, val_index) in enumerate(mskf.split(X, y)):        
        df_train.loc[val_index, "kfold"] = fold
    return df_train

def get_train_val_split_stats(df, num_folds, label_cols):
    counts = {}
    for fold in range(num_folds):
        y_train = df[df.kfold != fold][label_cols].values
        y_val = df[df.kfold == fold][label_cols].values
        counts[(fold, "train_count")] = Counter(
                                        str(combination) for row in get_combination_wise_output_matrix(y_train, order=1) 
                                        for combination in row
                                    )
        counts[(fold, "val_count")] = Counter(
                                        str(combination) for row in get_combination_wise_output_matrix(y_val, order=1) 
                                        for combination in row
                                    )
    # View distributions
    df_counts = pd.DataFrame(counts).T.fillna(0)
    df_counts.index.set_names(["fold", "counts"], inplace=True)
    for fold in range(num_folds):
        train_counts = df_counts.loc[(fold, "train_count"), :]
        val_counts = df_counts.loc[(fold, "val_count"), :]
        val_train_ratio = pd.Series({i: val_counts[i] / train_counts[i] for i in train_counts.index}, name=(fold, "val_train_ratio"))
        df_counts = df_counts.append(val_train_ratio)
    df_counts = df_counts.sort_index() 
    return df_counts        