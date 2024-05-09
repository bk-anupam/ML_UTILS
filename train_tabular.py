import numpy as np
import pandas as pd
import statistics
import optuna
import lightgbm as lgbm
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_log_error 
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

def train_fold_xgb(train_X, train_y, val_X, val_y, model_params, metric):
    fold_train_metric = None
    model = xgb.XGBRegressor(**model_params)
    model.fit(
            train_X, 
            train_y,
            #eval_metric=metric,
            eval_set=[(train_X, train_y), (val_X, val_y)],                        
            verbose=model_params["verbosity"]
        )
    val_y_pred = model.predict(val_X)
    fold_train_metric = get_eval_metric(metric, val_y, val_y_pred)
    return fold_train_metric, model, val_y_pred

def train_fold_lgbm(train_df, train_y, val_df, val_y, feature_col_names, metric,
                    params=None, callbacks=None):
    train_data = lgbm.Dataset(data=train_df[feature_col_names], label=train_y, feature_name="auto")    
    val_data = lgbm.Dataset(data=val_df[feature_col_names], label=val_y, feature_name="auto")
    model = lgbm.train(
        params=params,
        train_set=train_data,
        valid_sets=val_data
    )
    val_df = val_df[feature_col_names]
    val_preds = model.predict(val_df, num_iteration=model.best_iteration)
    fold_train_metric = get_eval_metric(metric, val_y, val_preds)
    return fold_train_metric, model, val_preds    

def run_training(model_name, df_train, target_col_name, feature_col_names=None, 
                 metric=Metrics.MAE, num_folds=5, single_fold=False, model_params=None, 
                 val_preds_col="val_preds", suppress_print=False):
    fold_metrics_model = []
    df_val_preds = pd.DataFrame()    
    for fold in range(num_folds):
        df_train_fold, df_val_fold = get_fold_df(df_train, fold)        
        train_X, train_y, val_X, val_y = get_train_val_nparray(df_train_fold, df_val_fold, feature_col_names, target_col_name)        
        if model_name == ModelName.LGBM:            
            fold_val_metric, fold_model, fold_val_preds = train_fold_lgbm(
                train_df=df_train_fold, 
                train_y=train_y, 
                val_df=df_val_fold, 
                val_y=val_y, 
                feature_col_names=feature_col_names,
                metric=metric,
                params=model_params
            )
        elif model_name == ModelName.XGBoost:
            fold_val_metric, fold_model, fold_val_preds = train_fold_xgb(
                train_X, train_y, val_X, val_y, model_params, metric=metric
            )
        else:
            model = get_model(model_name, model_params)            
            fold_val_metric, fold_model, fold_val_preds = train_fold(train_X, train_y, val_X, val_y, model, metric=metric)

        if not suppress_print:
            print(f"fold {fold} metric = {fold_val_metric}")
        df_val_fold[val_preds_col] = fold_val_preds
        df_val_preds = pd.concat([df_val_preds, df_val_fold], axis=0)
        fold_metrics_model.append((fold_val_metric, fold_model))
        if single_fold:
            break
    return fold_metrics_model, df_val_preds

def train_model(df, model_name, model_params, feature_col_names, target_col_name, metric=Metrics.MAE, num_folds=5, single_fold=False):
    val_preds_col = "val_preds"
    print(f"training {model_name}")
    if model_name == ModelName.L2_Ridge:
        val_preds_col = "l2_val_preds"

    fold_metrics_model, df_val_preds = run_training(
            model=model,
            df_train=df,
            target_col_name=target_col_name,
            feature_col_names=feature_col_names,
            metric=metric,            
            num_folds=num_folds,
            gb_params=model_params,
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
    cv = get_cv_score(df_val_preds, target_col_name, val_preds_col, metric)
    df_val_preds.to_csv(f"./data/df_val_preds_{model_name}.csv")
    print(f"Saved validation data predictions to df_val_preds_{model_name}.csv")
    print(f"{model_name} CV score = {cv}")
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