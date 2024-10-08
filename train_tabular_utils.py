import numpy as np
import pandas as pd
import statistics
import optuna
import lightgbm as lgbm
import xgboost as xgb
import catboost
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_log_error, r2_score, accuracy_score, roc_auc_score, f1_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from enums import ModelName, Scaler, Metrics
from sklearn import model_selection
from joblib import dump
from functools import partial

def get_metric_type(metric):
    if metric in [Metrics.MAE, Metrics.RMSLE, Metrics.MSE, Metrics.R2, Metrics.RMSE]:
        metric_type = "regression"
    else:
        metric_type = "classification"
    return metric_type

def get_fold_df(df, fold):
    df_fold_val = df[df.kfold == fold].reset_index(drop=True)
    df_fold_train = df[df.kfold != fold].reset_index(drop=True)
    return df_fold_train, df_fold_val    

def get_tree_model(model_name, metric_type, model_params, cat_features=None, callbacks=None):    
    model = None
    if model_name == ModelName.XGBoost and metric_type == "regression":    
        model = xgb.XGBRegressor(**model_params, callbacks=callbacks)
    elif model_name == ModelName.XGBoost and metric_type == "classification":
        model = xgb.XGBClassifier(**model_params)
    elif model_name == ModelName.CatBoost and metric_type == "regression":
        if cat_features is not None:
            model = catboost.CatBoostRegressor(**model_params, cat_features=cat_features)
        else:
            model = catboost.CatBoostRegressor(**model_params)
    elif model_name == ModelName.CatBoost and metric_type == "classification":
        model = catboost.CatBoostClassifier(**model_params)
    elif model_name == ModelName.LGBM and metric_type == "regression":
        model = lgbm.LGBMRegressor(**model_params)
    elif model_name == ModelName.LGBM and metric_type == "classification":
        model = lgbm.LGBMClassifier(**model_params)
    return model

def get_model(model_name, params, metric, random_state=42, cat_features=None, callbacks=None):
    model = None
    metric_type = get_metric_type(metric)
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
    elif model_name == ModelName.LogisticRegression:
        if params is not None:
            model = LogisticRegression(**params)
        else:
            model = LogisticRegression()
    elif model_name == ModelName.LinearRegression:
        model = LinearRegression()
    elif model_name == ModelName.RandomForest and metric_type == "regression":
        model = RandomForestRegressor(**params)
    elif model_name == ModelName.RandomForest and metric_type == "classification":
        model = RandomForestClassifier(**params)
    elif model_name == ModelName.GradientBoostingRegressor and metric_type == "regression":
        model = GradientBoostingRegressor(**params)
    elif model_name == ModelName.TabNetRegressor and metric_type == "regression":
        model = TabNetRegressor()
    else:
        model = get_tree_model(model_name, metric_type, model_params=params, cat_features=cat_features,
                               callbacks=callbacks)   
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
    elif metric == Metrics.RMSE:
        fold_train_metric = np.sqrt(mean_squared_error(val_y, val_y_pred))
    elif metric == Metrics.ACCURACY:        
        fold_train_metric = accuracy_score(val_y, val_y_pred)
    elif metric ==  Metrics.AUC:
        fold_train_metric = roc_auc_score(val_y, val_y_pred)
    elif metric == Metrics.F1:        
        fold_train_metric = f1_score(val_y, val_y_pred)    
    elif metric == Metrics.R2:
        fold_train_metric = r2_score(val_y, val_y_pred)
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

def get_metric_stats(fold_metrics):
    mean_metric = np.mean(fold_metrics)
    std_metric = np.std(fold_metrics)
    return mean_metric, std_metric

def train_fold_xgb_cb(model_name, train_X, train_y, val_X, val_y, model_params, metric, transform_target=False):
    fold_train_metric = None
    metric_type = get_metric_type(metric)
    model = get_tree_model(model_name, metric_type, model_params) 
    verbose = None
    if model_name == ModelName.CatBoost:       
        verbose = model_params["verbose"]
    elif model_name == ModelName.XGBoost:
        verbose = model_params["verbosity"]
    model.fit(
            train_X, 
            train_y,            
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
    
def fit_fold_model(model_name, model_params, fold_model, train_X, train_y, val_X, val_y, 
                   cat_features=None, callbacks=None):    
    if model_name == ModelName.CatBoost:       
        verbose = model_params["verbose"]
        fold_model.fit(X=train_X, y=train_y, eval_set=[(val_X, val_y)], verbose=verbose, 
                       cat_features=cat_features, callbacks=callbacks)
    elif model_name == ModelName.XGBoost:
        verbose = model_params["verbosity"]
        fold_model.fit(train_X, train_y, eval_set=[(val_X, val_y)], verbose=verbose)
    elif model_name == ModelName.LGBM:
        fold_model.fit(train_X, train_y, eval_set=[(val_X, val_y)], callbacks=callbacks)
    else:
        fold_model.fit(train_X, train_y)
    return fold_model, val_X, val_y

def get_fold_val_preds(model, val_X, val_y, metric, df_fold_val_preds, num_classes=None):
    metric_type = get_metric_type(metric)
    if metric_type == "regression":
        val_preds = model.predict(val_X)
        df_fold_val_preds['oof_preds'] = val_preds
        fold_val_metric = get_eval_metric(metric, val_y, val_preds)
    else:
        val_preds_proba = model.predict_proba(val_X)
        if metric in [Metrics.F1, Metrics.ACCURACY]:
            val_preds = val_preds_proba.argmax(axis=1)
            fold_val_metric = get_eval_metric(metric, val_y, val_preds)
            df_fold_val_preds['oof_preds'] = val_preds
        elif metric == Metrics.AUC:
            # for binary classification case            
            fold_val_metric = get_eval_metric(metric, val_y, val_preds_proba[:,1])
            df_fold_val_preds['oof_preds'] = val_preds_proba[:,1]            
        if num_classes is not None:
            for i in range(num_classes):
                df_fold_val_preds[f"oof_preds_proba_{i}"] = val_preds_proba[:,i]
    return fold_val_metric, df_fold_val_preds

def train_validate_fold(fold, model_name, model_params, preprocessor, df, feature_cols, 
                        metric, target_col_name, num_classes=None):
    fold_model = get_model(model_name=model_name, params=model_params, metric=metric)
    df_train_fold, df_val_fold = get_fold_df(df, fold)
    train_X, train_y, val_X, val_y = get_train_val_nparray(df_train_fold, df_val_fold, feature_cols, target_col_name)
    if preprocessor is not None:
        train_X = preprocessor.fit_transform(train_X)
        val_X = preprocessor.transform(val_X)
    fit_fold_model(model_name, model_params, fold_model, train_X, train_y, val_X, val_y)
    df_fold_val_preds = df_val_fold[['kfold', target_col_name]]
    fold_val_metric, df_fold_val_preds = get_fold_val_preds(fold_model, val_X, val_y, metric, 
                                                            df_fold_val_preds, num_classes=num_classes)
    return fold_val_metric

def impute_missing_values(df_train_fold, df_val_fold, imputation_config):
    """
    Imputes missing values in the given training and validation dataframes.

    Args:
        df_train_fold (pd.DataFrame): The training dataframe with missing values.
        df_val_fold (pd.DataFrame): The validation dataframe with missing values.
        imputation_config (dict): A dictionary containing column names as keys and imputers as values.

    Returns:
        None
    """
    for column, imputer in imputation_config.items():
        imputer.fit(df_train_fold[[column]])
        df_train_fold[column] = imputer.transform(df_train_fold[[column]])
        df_val_fold[column] = imputer.transform(df_val_fold[[column]])

def train_and_validate(model_name, model_params, preprocessor, df, feature_cols, 
                       target_col_name, metric, single_fold=False, num_folds=5, 
                       suppress_print=False, num_classes=None, imputation_config=None, cat_features=None, 
                       cat_encoders=None, callbacks=None):    
    df_oof_preds = pd.DataFrame()
    fold_metrics_model = []    
    for fold in range(num_folds):
        fold_model = get_model(model_name=model_name, params=model_params, metric=metric, 
                               cat_features=cat_features,
                               callbacks=callbacks)
        df_train_fold, df_val_fold = get_fold_df(df, fold)
        if imputation_config is not None:
            impute_missing_values(df_train_fold, df_val_fold, imputation_config)
        if cat_encoders is not None:
            for col, encoders in cat_encoders.items():    
                for encoder in encoders:
                    df_train_fold[[col]] = encoder.fit_transform(df_train_fold[[col]], df_train_fold[target_col_name])
                    df_val_fold[[col]] = encoder.transform(df_val_fold[[col]])
        train_X, train_y, val_X, val_y = get_train_val_nparray(df_train_fold, df_val_fold, feature_cols, target_col_name)
        if preprocessor is not None:
            train_X = preprocessor.fit_transform(train_X)
            val_X = preprocessor.transform(val_X)
        fold_model, val_X, val_y = fit_fold_model(
            model_name, model_params, fold_model, train_X, train_y, val_X, val_y, 
            cat_features=cat_features, 
            callbacks=callbacks
        )
        df_fold_val_preds = df_val_fold[['kfold', target_col_name]]
        fold_val_metric, df_fold_val_preds = get_fold_val_preds(fold_model, val_X, val_y, metric, 
                                                                df_fold_val_preds, num_classes=num_classes)
        if not suppress_print:        
            print(f"Fold {fold} - {model_name} - {metric} : {fold_val_metric}")                
        df_oof_preds = pd.concat([df_oof_preds, df_fold_val_preds], axis=0)
        fold_metrics_model.append((fold_val_metric, fold_model))
        if single_fold:
            break
    cv = get_eval_metric(metric, df_oof_preds[target_col_name], df_oof_preds['oof_preds'] )
    metrics = [item[0] for item in fold_metrics_model]
    mean_metric, std_metric = get_metric_stats(metrics)
    if not suppress_print:    
        print(f"{model_name} metric={metric} CV score = {cv}")            
        print(f"{model_name} Mean {metric} = {mean_metric}, std = {std_metric}")
    return fold_metrics_model, df_oof_preds, preprocessor    

def get_cv_score(fold_metrics_model, model_name, metric, df_oof_preds, target_col_name):
    metrics = [item[0] for item in fold_metrics_model]
    for fold, fold_metric in enumerate(metrics):
        print(f"Fold {fold} - {model_name} - {metric} : {fold_metric}")
    cv = get_eval_metric(metric, df_oof_preds[target_col_name], df_oof_preds['oof_preds'] )
    print(f"{model_name} metric={metric} CV score = {cv}")    
    mean_metric, std_metric = get_metric_stats(metrics)    
    print(f"{model_name} Mean {metric} = {mean_metric}, std = {std_metric}")

def persist(model_name, fold_metrics_model, df_oof_preds, persist_model=False, output_path=""):    
    fold_models = [item[1] for item in fold_metrics_model]    
    if persist_model:
        for index, model in enumerate(fold_models):
            fold_model_name = output_path + f"{model_name}_{index}.joblib"        
            dump(model, fold_model_name)
            print(f"saved {fold_model_name}")    
    df_oof_preds.to_csv(output_path + f"df_val_preds_{model_name}.csv")
    print(f"Saved validation data predictions to df_val_preds_{model_name}.csv")  

def get_fold_test_preds(fold_metrics_model, test_X, num_folds):
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
        fold_test_preds = model.predict(test_X)            
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

def get_test_preds(fold_metrics_model, df_test, feature_cols, preprocessor=None, num_folds=5):
    # For each fold, get the test predictions using corresponding fold model
    test_X = df_test.loc[:, feature_cols]
    if preprocessor is not None:
        test_X = preprocessor.transform(test_X)
    df_fold_test_preds = get_fold_test_preds(fold_metrics_model, test_X=test_X, num_folds = num_folds)
    fold_metrics = [item[0] for item in fold_metrics_model]
    # normalize the fold weights
    fold_weights = fold_metrics / np.sum(fold_metrics)
    # Combine fold predictions using simple averaging    
    df_fold_test_preds["test_preds"] = combine_fold_test_preds(df_fold_test_preds, fold_weights=None)
    return df_fold_test_preds    

def get_test_preds_clf(fold_metrics_model, df_test, feature_cols, preprocessor=None, num_folds=5):
    test_X = df_test.loc[:, feature_cols]
    if preprocessor is not None:
        test_X = preprocessor.transform(test_X)
    print(f"test_X shape: {test_X.shape}")
    test_preds_proba = []
    for fold in range(num_folds):
        model = fold_metrics_model[fold][1]
        fold_test_preds_proba = model.predict_proba(test_X)
        test_preds_proba.append(fold_test_preds_proba)
    fold_metrics = [item[0] for item in fold_metrics_model]
    # normalize the fold weights
    fold_weights = fold_metrics / np.sum(fold_metrics)
    test_preds_proba = np.average(test_preds_proba, axis=0, weights=fold_weights)
    print(f"test_preds_proba shape: {test_preds_proba.shape}")
    test_preds = np.argmax(test_preds_proba, axis=1)
    print(f"test_preds shape: {test_preds.shape}")
    return test_preds_proba, test_preds

def get_candidate_features(df_feature_imp, feature_selection_method, base_features_count=None):
    if feature_selection_method == "forward":
        # all remanining features are candidate features (get rows from df_feature_imp that are not in base_features)
        if base_features_count is None:
            candidate_features = df_feature_imp.iloc[:]["f_name"].values
        else:
            candidate_features = df_feature_imp.iloc[base_features_count:, :]["f_name"].values        
    else:
        # for backward feature selection features from last to number of base features are candidate features
        if base_features_count is None:
            candidate_features = df_feature_imp.iloc[:]["f_name"].values
        else:
            candidate_features = df_feature_imp.iloc[base_features_count:]["f_name"].values
        candidate_features = candidate_features[::-1]
    return candidate_features    
    
def feature_selection(feature_selection_method, candidate_features, selected_features,
    model_params, df_train, preprocessor, imputation_config, cat_encoders, cat_features, model_name,
    target_col_name, metric, num_folds=5, single_fold=True, best_score_initial=0.0, metric_minimize=True):    
    unselected_features = []
    best_score = best_score_initial
    print(f"FEATURE SELECTION METHOD = {feature_selection_method}")
    for feature in candidate_features:
        if feature_selection_method == "forward":
            cur_features = np.append(selected_features, feature)
        else:
            cur_features = np.delete(selected_features, np.where(selected_features == feature))
        fold_metrics_model, _, _ = train_and_validate(
            model_name=model_name,
            model_params=model_params,
            preprocessor=preprocessor,
            df=df_train,
            feature_cols=cur_features,
            target_col_name=target_col_name,
            metric=metric,
            single_fold=single_fold,
            num_folds=num_folds,
            suppress_print=True,
            imputation_config=imputation_config,
            cat_features=cat_features,
            cat_encoders=cat_encoders
        )
        fold_metrics = [item[0] for item in fold_metrics_model]
        cur_val_score = statistics.mean(fold_metrics)        
        is_better_score = cur_val_score < best_score if metric_minimize else cur_val_score > best_score
        if is_better_score:
            print(f"feature selected = {feature} | cur_val_score = {cur_val_score} | best_score = {best_score}")
            if feature_selection_method == "forward":
                selected_features = np.append(selected_features, feature)
            else:
                selected_features = np.delete(selected_features, np.where(selected_features == feature))
            best_score = cur_val_score
        else:
            print(f"feature not selected = {feature} | cur_val_score = {cur_val_score} | best_score = {best_score}")
            unselected_features.append(feature)
    return selected_features, unselected_features