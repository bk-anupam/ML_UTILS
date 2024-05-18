import numpy as np
import pandas as pd
import statistics
import optuna
import lightgbm as lgbm
import xgboost as xgb
import catboost
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_log_error, mean_squared_error 
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from enums import ModelName, Scaler, Metrics
from sklearn import model_selection
from joblib import dump
from functools import partial

def get_fold_df(df, fold):
    df_fold_val = df[df.kfold == fold].reset_index(drop=True)
    df_fold_train = df[df.kfold != fold].reset_index(drop=True)
    return df_fold_train, df_fold_val    

def get_model(model_name, params, random_state=42):
    model = None
    if model_name in [ModelName.Ridge, ModelName.L2_Ridge]:
        if params is not None:
            model = Ridge(alpha = params["alpha"], random_state=random_state)
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

def get_scaler(scaler):
    if scaler == Scaler.RobustScaler:
        scaler = RobustScaler()
    elif scaler == Scaler.MinMaxScaler:
        scaler = MinMaxScaler()
    elif scaler == Scaler.StandardScaler: 
        scaler = StandardScaler()
    else: 
        scaler = None
    return scaler

def normalize_features(df, scaler, features_to_normalize):
    scaler = get_scaler(scaler)
    if scaler is not None:
        df[features_to_normalize] = scaler.fit_transform(df[features_to_normalize].astype(np.float32))
    return df

def get_eval_metric(metric, val_y, val_y_pred):
    fold_train_metric = None
    if metric == Metrics.MAE:
        fold_train_metric = mean_absolute_error(val_y, val_y_pred)
    elif metric == Metrics.RMSLE:                        
        # we can't use negative predictions for RMSLE
        val_y_pred = [item if item > 0 else 0 for item in val_y_pred]
        fold_train_metric = np.sqrt(mean_squared_log_error(val_y, val_y_pred))
    return fold_train_metric

def train_fold(train_X, train_y, val_X, val_y, model, metric=Metrics.MAE):            
    fold_train_metric = None
    model.fit(train_X, train_y.ravel())
    val_y_pred = model.predict(val_X)
    fold_train_metric = get_eval_metric(metric, val_y, val_y_pred)
    return fold_train_metric, model, val_y_pred

def get_tree_model(model_name, metric, model_params):
    metric_type = None
    model = None
    if metric in [Metrics.MAE, Metrics.RMSLE, Metrics.MSE, Metrics.R2]:
        metric_type = "regression"
    else:
        metric_type = "classification"
    if model_name == ModelName.XGBoost and metric_type == "regression":    
        model = xgb.XGBRegressor(**model_params)
    elif model_name == ModelName.XGBoost and metric_type == "classification":
        model = xgb.XGBClassifier(**model_params)
    elif model_name == ModelName.CatBoost and metric_type == "regression":
        model = catboost.CatBoostRegressor(**model_params)
    elif model_name == ModelName.CatBoost and metric_type == "classification":
        model = catboost.CatBoostClassifier(**model_params)
    return model

def get_metric_stats(fold_metrics):
    mean_metric = np.mean(fold_metrics)
    std_metric = np.std(fold_metrics)
    return mean_metric, std_metric

def train_fold_xgb_cb(model_name, train_X, train_y, val_X, val_y, model_params, metric, transform_target=False):
    fold_train_metric = None
    model = get_tree_model(model_name, metric, model_params) 
    verbose = None
    if model_name == ModelName.CatBoost:       
        verbose = model_params["verbose"]
    elif model_name == ModelName.XGBoost:
        verbose = model_params["verbosity"]
    model.fit(
            train_X, 
            train_y,
            #eval_metric=metric,
            eval_set=[(train_X, train_y), (val_X, val_y)],                        
            verbose=verbose
        )
    val_y_pred = model.predict(val_X)
    if metric == Metrics.RMSLE and transform_target:
        # Since we have trained on np.log1p(y) instead of y, we need to reverse the transformation to extract the actual predictions        
        val_y_pred = np.expm1(val_y_pred)
    fold_train_metric = get_eval_metric(metric, val_y, val_y_pred)
    return fold_train_metric, model, val_y_pred

def train_fold_lgbm(train_df, train_y, val_df, val_y, feature_col_names, metric,
                    params=None, transform_target=False, callbacks=None):
    train_data = lgbm.Dataset(data=train_df[feature_col_names], label=train_y, feature_name="auto")    
    val_data = lgbm.Dataset(data=val_df[feature_col_names], label=val_y, feature_name="auto")
    model = lgbm.train(
        params=params,
        train_set=train_data,
        valid_sets=val_data
    )
    val_df = val_df[feature_col_names]
    val_preds = model.predict(val_df, num_iteration=model.best_iteration)
    if metric == Metrics.RMSLE and transform_target:
        # Since we have trained on np.log1p(y) instead of y, we need to reverse the transformation to extract the actual predictions
        val_preds = np.expm1(val_preds)
    fold_train_metric = get_eval_metric(metric, val_y, val_preds)
    return fold_train_metric, model, val_preds    

def run_training(model_name, df_train, target_col_name, feature_col_names=None, 
                 metric=Metrics.MAE, num_folds=5, single_fold=False, model_params=None, 
                 val_preds_col="val_preds", suppress_print=False, transform_target=False):
    fold_metrics_model = []
    df_val_preds = pd.DataFrame()    
    for fold in range(num_folds):
        df_train_fold, df_val_fold = get_fold_df(df_train, fold)        
        train_X, train_y, val_X, val_y = get_train_val_nparray(df_train_fold, df_val_fold, feature_col_names, target_col_name)        
        # To train on RMSLE objective instead of RMSE we need to transform the target values
        if metric == Metrics.RMSLE and transform_target:            
            train_y = np.log1p(train_y)            
        if model_name == ModelName.LGBM:            
            fold_val_metric, fold_model, fold_val_preds = train_fold_lgbm(
                train_df=df_train_fold, 
                train_y=train_y, 
                val_df=df_val_fold, 
                val_y=val_y, 
                feature_col_names=feature_col_names,
                metric=metric,
                params=model_params,
                transform_target=transform_target
            )
        elif model_name in [ModelName.XGBoost, ModelName.CatBoost]:
            fold_val_metric, fold_model, fold_val_preds = train_fold_xgb_cb(
                model_name, train_X, train_y, val_X, val_y, model_params, metric=metric, transform_target=transform_target
            )
        else:
            model = get_model(model_name, model_params)            
            fold_val_metric, fold_model, fold_val_preds = train_fold(train_X, train_y, val_X, val_y, model, metric=metric)

        if not suppress_print:
            print(f"Fold {fold} - {model_name} - {metric} : {fold_val_metric}")
        df_val_fold[val_preds_col] = fold_val_preds
        df_val_preds = pd.concat([df_val_preds, df_val_fold], axis=0)
        fold_metrics_model.append((fold_val_metric, fold_model))
        if single_fold:
            break
    return fold_metrics_model, df_val_preds
    
def train_model(df, model_name, model_params, feature_col_names, target_col_name, 
                metric=Metrics.MAE, num_folds=5, single_fold=False, persist_model=False,
                output_path="", transform_target=False):
    val_preds_col = "val_preds"
    print(f"training {model_name}")
    if model_name == ModelName.L2_Ridge:
        val_preds_col = "l2_val_preds"

    fold_metrics_model, df_val_preds = run_training(
            model_name=model_name,
            df_train=df,
            target_col_name=target_col_name,
            feature_col_names=feature_col_names,
            metric=metric,            
            num_folds=num_folds,
            model_params=model_params,
            val_preds_col=val_preds_col,
            single_fold=single_fold,
            transform_target=transform_target
        )        
    metrics = [item[0] for item in fold_metrics_model]
    fold_models = [item[1] for item in fold_metrics_model]    
    if persist_model:
        for index, model in enumerate(fold_models):
            fold_model_name = output_path + f"{model_name}_{index}.joblib"        
            dump(model, fold_model_name)
            print(f"saved {fold_model_name}")
    cv = get_cv_score(df_val_preds, target_col_name, val_preds_col, metric)
    mean_metric, std_metric = get_metric_stats(metrics)
    df_val_preds.to_csv(output_path + f"df_val_preds_{model_name}.csv")
    print(f"Saved validation data predictions to df_val_preds_{model_name}.csv")
    print(f"{model_name} CV score = {cv}")
    print(f"{model_name} Mean {metric} = {mean_metric}, std = {std_metric}")
    return fold_metrics_model

def get_cv_score(df_val_preds, target_col_name, val_preds_col, metric):
    y_true_cv = df_val_preds[target_col_name]
    y_pred_cv = df_val_preds[val_preds_col]
    cv_score = None
    if metric == Metrics.MAE:
        cv_score = mean_absolute_error(y_true_cv, y_pred_cv)
    elif metric == Metrics.RMSLE:
        cv_score = np.sqrt(mean_squared_log_error(y_true_cv, y_pred_cv))
    return cv_score

def get_fold_test_preds(fold_metrics_model, df_test, feature_cols, num_folds):
    """
    Generate predictions for each fold on the test dataset.

    Args:
        fold_metrics_model (list): A list of tuples containing the metrics and models for each fold.
        df_test (pandas.DataFrame): The test dataset.
        feature_cols (list): The list of feature column names.
        num_folds (int): The number of folds.

    Returns:
        pandas.DataFrame: A dataframe containing the predictions for each fold. The column names are in the format "fold_{fold}_test_preds".

    """
    fold_test_preds_dict = {}
    for fold in range(num_folds):
        model = fold_metrics_model[fold][1]    
        test_df = df_test[feature_cols]             
        fold_test_preds = model.predict(test_df)            
        pred_col_name = f"fold_{fold}_test_preds"
        fold_test_preds_dict[pred_col_name] = fold_test_preds 
    df_fold_test_preds = pd.DataFrame(fold_test_preds_dict)
    return df_fold_test_preds

def combine_fold_test_preds(df_fold_test_preds, fold_weights = None):
    """
    Combine the predictions from multiple folds into a single prediction.

    Args:
        df_fold_test_preds (pandas.DataFrame): A DataFrame containing the predictions from multiple folds.
        fold_weights (list, optional): A list of weights to be applied to each fold. Defaults to None.

    Returns:
        pandas.Series: A Series containing the combined prediction for each row in the DataFrame.

    Raises:
        ValueError: If the length of fold_weights does not match the number of columns in df_fold_test_preds.

    """
    if fold_weights is None:
        fold_weights = [1] * len(df_fold_test_preds.columns)
    return df_fold_test_preds.apply(lambda x: np.average(x, weights=fold_weights), axis=1)
