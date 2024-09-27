import optuna
import statistics
from functools import partial
import train_tabular_utils as tt
import enums
from enums import ModelName
from optuna.integration import XGBoostPruningCallback, LightGBMPruningCallback, CatBoostPruningCallback
from optuna.pruners import MedianPruner

def get_params_from_config(trial, param_ranges):
    params_dynamic = {}
    for param, range_config in param_ranges.items():
        if range_config['type'] == 'loguniform' :
            params_dynamic[param] = trial.suggest_loguniform(param, range_config['min_value'], range_config['max_value'])
        elif range_config['type'] == 'int':
            if 'step' in range_config:
                params_dynamic[param] = trial.suggest_int(param, range_config['min_value'], range_config['max_value'], 
                                                          step=range_config['step'])
            else:
                params_dynamic[param] = trial.suggest_int(param, range_config['min_value'], range_config['max_value'])
        else:
            # check if key 'log' exists
            if 'log' in range_config:
                params_dynamic[param] = trial.suggest_float(param, range_config['min_value'], range_config['max_value'], 
                                                            log=range_config['log'])
            else:
                if 'step' in range_config:
                    params_dynamic[param] = trial.suggest_float(param, range_config['min_value'], range_config['max_value'], 
                                                                step=range_config['step'])
                else:
                    params_dynamic[param] = trial.suggest_float(param, range_config['min_value'], range_config['max_value'])
    return params_dynamic    

def get_model_tuning_params(trial, model_name, static_params, param_ranges, tuning_level=None, 
                            level_params_totune=None, params_defaults=None, best_params_level1=None, 
                            best_params_level2=None):
    model_static_params = static_params[model_name]
    params_with_default = None    
    if tuning_level is not None:
        params_with_no_defaults = []
        params_totune = level_params_totune[str(tuning_level)]        
        param_ranges = {k: v for k, v in param_ranges.items() if k in params_totune}
        # add items in params_totune list to params_with_no_defaults
        params_with_no_defaults = params_totune.copy()        
        if best_params_level1 is not None:            
            params_with_no_defaults.extend(list(best_params_level1.keys()))
        if best_params_level2 is not None:
            params_with_no_defaults.extend(list(best_params_level2.keys()))
        params_with_default = {k: v for k, v in params_defaults.items() if k not in params_with_no_defaults}

    params_from_config = get_params_from_config(trial, param_ranges)
    model_params = {**model_static_params, **params_from_config}
    # check if param_with_default is not None or empty
    if params_with_default is not None and len(params_with_default) > 0:
        model_params = {**model_params, **params_with_default}
    if best_params_level1 is not None:
        model_params = {**model_params, **best_params_level1}
    if best_params_level2 is not None:
        model_params = {**model_params, **best_params_level2}
    return model_params
    
def hyperparams_tuning_objective(trial, model_name, preprocessor, df,  
                                 feature_cols, metric, target_col_name, single_fold=False, num_folds=5,
                                 imputation_config=None, cat_features=None, cat_encoders=None,
                                 tuning_level=None, level_params_totune=None, params_defaults=None, 
                                 best_params_level1=None, best_params_level2=None,
                                 param_ranges=None, static_params=None):               
    model_params = get_model_tuning_params(
        trial, 
        model_name=model_name, 
        static_params=static_params,
        param_ranges=param_ranges, 
        tuning_level=tuning_level,
        level_params_totune=level_params_totune,
        params_defaults=params_defaults,
        best_params_level1=best_params_level1, 
        best_params_level2=best_params_level2
    ) 
    callbacks = None
    if model_name == ModelName.XGBoost:
        pruning_callback = XGBoostPruningCallback(trial, f"validation_0-{metric.lower()}")
        callbacks = [pruning_callback]
    elif model_name == ModelName.LGBM:
        pruning_callback = LightGBMPruningCallback(trial, metric.lower())
        callbacks = [pruning_callback]
    elif model_name == ModelName.CatBoost and static_params[ModelName.CatBoost]['task_type'] != 'GPU':
        pruning_callback = CatBoostPruningCallback(trial, metric)
        callbacks = [pruning_callback]
    fold_metrics_model, _, _ = tt.train_and_validate(
                                        model_name=model_name,
                                        model_params=model_params,
                                        preprocessor=preprocessor,
                                        df=df,
                                        feature_cols=feature_cols,
                                        target_col_name=target_col_name,
                                        metric=metric,
                                        single_fold=single_fold,
                                        num_folds=num_folds,
                                        suppress_print=True,
                                        imputation_config=imputation_config,
                                        cat_features=cat_features,
                                        cat_encoders=cat_encoders,
                                        callbacks=callbacks
                                    )
    fold_metrics = [x[0] for x in fold_metrics_model]
    mean_metric = statistics.mean(fold_metrics)                
    return mean_metric

def optimize_trial(num_trials, model_params_tuning_level1_obj_partial, study, tuning_level=None):
    study.optimize(model_params_tuning_level1_obj_partial, n_trials=num_trials)
    best_trial = study.best_trial
    tuning_level_str = f"level {tuning_level}" if tuning_level is not None else ""
    print(f"Best trial {tuning_level_str}: number = {best_trial.number}, value = {best_trial.value}, params = {best_trial.params}")
    return best_trial.params

def tune_model_params(study_name, study_direction, num_trials, model_name, 
                      preprocessor, df, feature_cols, metric, target_col_name, 
                      single_fold=False, num_folds=5, imputation_config=None,
                      cat_features=None, cat_encoders=None, stepwise=False, param_ranges=None,
                      level_params_totune=None, params_defaults=None, static_params=None):
    def create_study(level=None):
        if stepwise and level:
            return optuna.create_study(
                direction=study_direction, 
                study_name=f"{study_name}_level{level}",
                pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=30))
        else:
            return optuna.create_study(
                direction=study_direction, 
                study_name=study_name,
                pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=30))

    def create_partial(level=None, best_params=None):
        return partial(
            hyperparams_tuning_objective,
            model_name=model_name,
            preprocessor=preprocessor,
            df=df,
            feature_cols=feature_cols,
            metric=metric,
            target_col_name=target_col_name,
            single_fold=single_fold,
            num_folds=num_folds,
            imputation_config=imputation_config,
            cat_features=cat_features,
            cat_encoders=cat_encoders,            
            level_params_totune=level_params_totune,
            params_defaults=params_defaults,
            static_params=static_params,
            param_ranges=param_ranges,
            tuning_level=level,            
            best_params_level1=best_params[0] if best_params else None,
            best_params_level2=best_params[1] if best_params and len(best_params) > 1 else None
        )    
    if stepwise:
        best_params = []
        for level in range(1, 4):
            study = create_study(level)
            partial_obj = create_partial(level, best_params)
            best_params.append(optimize_trial(num_trials, partial_obj, study, level))
        # best_params list contains the best params for each level in the order of [level 1, level 2, level 3], params are
        # in form of dictionary. Create a single param dictionary by concatenating the best params from each level
        best_params_combined = {k: v for d in best_params for k, v in d.items()}
        return best_params_combined
    else:
        study = create_study()
        partial_obj = create_partial()
        return optimize_trial(num_trials, partial_obj, study)