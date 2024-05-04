import numpy as np
import pandas as pd
import lightgbm as lgbm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from model_name import ModelName
from sklearn import model_selection
from joblib import dump


# split the training dataframe into kfolds for cross validation. We do this before any processing is done on the data.
def kfold_dataframe(df, target_col_name, num_folds=5, n_bins=None):
    # we create a new column called kfold and fill it with -1
    df["kfold"] = -1
    # randomize of shuffle the rows of dataframe before splitting is done
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    # If target is continuous, we need to create bins first before performing stratification
    if n_bins is not None:
        df['target_grp'] = pd.cut(df[target_col_name], n_bins, labels=False)
        y = df['target_grp'].values
    else:
        # get the target data
        y = df[target_col_name].values
    skf = model_selection.StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    for fold, (train_index, val_index) in enumerate(skf.split(X=df, y=y)):
        df.loc[val_index, "kfold"] = fold
    return df   

def get_fold_df(df, fold):
    df_fold_val = df[df.kfold == fold].reset_index(drop=True)
    df_fold_train = df[df.kfold != fold].reset_index(drop=True)
    return df_fold_train, df_fold_val    

def get_model(model_name, params, random_state=42):
    model = None
    if model_name in [ModelName.Ridge, ModelName.L2_Ridge]:
        if params is not None:
            model = Ridge(alpha = params["alpha"])
        else:
            model = Ridge()
    elif model_name == ModelName.Lasso:
        if params is not None:
            model = Lasso(alpha = params["alpha"])
        else:
            model = Lasso()
    elif model_name == ModelName.LinearRegression:
        model = LinearRegression()
    elif model_name == ModelName.RandomForest:
        model = RandomForestRegressor(
                n_estimators=params["n_estimators"],                 
                max_depth=params["max_depth"],
                min_samples_leaf=params["min_samples_leaf"],
                min_samples_split=params["min_samples_split"],
                max_features=params["max_features"],
                random_state=random_state,
                n_jobs=-1
            )
    elif model_name == ModelName.GradientBoostingRegressor:
        model = GradientBoostingRegressor(                
                n_estimators=params["n_estimators"],                 
                max_depth=params["max_depth"],
                min_samples_leaf=params["min_samples_leaf"],                
                min_samples_split=params["min_samples_split"],                
                max_features=params["max_features"],                                
                subsample=params["subsample"],
                learning_rate=params["learning_rate"],
                random_state=random_state,                
                verbose=params["verbose"],
                n_iter_no_change=params["n_iter_no_change"])   
    return model                

def get_train_val_nparray(df_train_fold, df_val_fold, feature_col_names, target_col_name):
    train_X = df_train_fold.loc[:, feature_col_names]
    train_y = df_train_fold.loc[:, target_col_name]
    val_X = df_val_fold.loc[:, feature_col_names]
    val_y = df_val_fold.loc[:, target_col_name]
    return train_X, train_y, val_X, val_y

def train_fold(train_X, train_y, val_X, val_y, model, normalize=True):        
    scaler = StandardScaler()
    if normalize:
        train_X = scaler.fit_transform(train_X.astype(np.float32))
        val_X = scaler.fit_transform(val_X.astype(np.float32))        
    model.fit(train_X, train_y.ravel())
    val_y_pred = model.predict(val_X)
    return mean_absolute_error(val_y, val_y_pred), model, val_y_pred

def train_fold_lgbm(train_df, train_y, val_df, val_y, feature_col_names, params=None, callbacks=None):
    train_data = lgbm.Dataset(data=train_df[feature_col_names], label=train_y, feature_name="auto")    
    val_data = lgbm.Dataset(data=val_df[feature_col_names], label=val_y, feature_name="auto")
    model = lgbm.train(
        params=params,
        train_set=train_data,
        valid_sets=val_data,
        verbose_eval=-1
    )
    val_df = val_df[feature_col_names]
    val_preds = model.predict(val_df, num_iteration=model.best_iteration)
    mae = mean_absolute_error(val_y, val_preds)
    return mae, model, val_preds    

def run_training(model, df_train, target_col_name, num_folds=5, 
                 single_fold=False, feature_col_names=None, gb_params=None, val_preds_col="val_preds"):
    fold_metrics_model = []
    df_val_preds = pd.DataFrame()
    normalize_data = True
    for fold in range(num_folds):
        df_train_fold, df_val_fold = get_fold_df(df_train, fold)        
        train_X, train_y, val_X, val_y = get_train_val_nparray(df_train_fold, df_val_fold, feature_col_names, target_col_name)        
        if gb_params is None:
            if val_preds_col == "l2_val_preds":
                normalize_data = False            
            fold_val_metric, fold_model, fold_val_preds = train_fold(train_X, train_y, val_X, val_y, model, normalize=normalize_data)
        else:            
            fold_val_metric, fold_model, fold_val_preds = train_fold_lgbm(
                train_df=df_train_fold, 
                train_y=train_y, 
                val_df=df_val_fold, 
                val_y=val_y, 
                feature_col_names=feature_col_names,
                params=gb_params
            )
        print(f"fold {fold} metric = {fold_val_metric}")
        df_val_fold[val_preds_col] = fold_val_preds
        df_val_preds = pd.concat([df_val_preds, df_val_fold], axis=0)
        fold_metrics_model.append((fold_val_metric, fold_model))
        if single_fold:
            break
    return fold_metrics_model, df_val_preds

def train_model(df, model_name, model_params, feature_col_names, target_col_name, num_folds, single_fold=False):
    val_preds_col = "val_preds"
    print(f"training {model_name}")
    if model_name == ModelName.LGBM:
        model = None        
        fold_metrics_model, df_val_preds = run_training(
            model=model,
            df_train=df,
            target_col_name=target_col_name,
            feature_col_names=feature_col_names,
            num_folds=num_folds,
            gb_params=model_params,
            single_fold=single_fold
        )
    else:
        model = get_model(model_name, model_params)        
        if model_name == ModelName.L2_Ridge:
            val_preds_col = "l2_val_preds"
        fold_metrics_model, df_val_preds = run_training(
            model=model,
            df_train=df,
            target_col_name=target_col_name,
            feature_col_names=feature_col_names,
            num_folds=num_folds,
            val_preds_col=val_preds_col,
            single_fold=single_fold
        )
    metrics = [item[0] for item in fold_metrics_model]
    fold_models = [item[1] for item in fold_metrics_model]
    # save fold models to pickle files
    for index, model in enumerate(fold_models):
        fold_model_name = f"./models/{model_name}_{index}.joblib"        
        dump(model, fold_model_name)
        print(f"saved {fold_model_name}")
    cv = get_cv_score(df_val_preds, target_col_name, val_preds_col)
    df_val_preds.to_csv(f"./data/df_val_preds_{model_name}.csv")
    print(f"Saved validation data predictions to df_val_preds_{model_name}.csv")
    print(f"{model_name} CV score = {cv}")
    return fold_metrics_model

def get_cv_score(df_val_preds, target_col_name, val_preds_col):
    y_true_cv = df_val_preds[target_col_name]
    y_pred_cv = df_val_preds[val_preds_col]
    return mean_absolute_error(y_true_cv, y_pred_cv)