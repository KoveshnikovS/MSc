import argparse
import joblib
import pandas as pd
import numpy as np
from typing import Text
import yaml
from utils.logs import get_logger
from utils.training import train
from utils.DL_training import DL_train
from pathlib import Path
import json

def train_model(config_path: Text) -> None:
    """### Train model
    Controls the training, passing the datasets to the needed method, which it envokes (DL_training.py or training.py).

    #### Parameters:
    - config_path {Text}: path to config (params.yaml file)

    #### Returns:
    - model {file}: saves trained model to /Saved models/model
    """

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    logger = get_logger('TRAIN', log_level=config['base']['log_level'])

    logger.info('Get estimator name')
    estimator_name = config['train']['estimator_name']
    logger.info(f'Estimator: {estimator_name}')

    logger.info('Load train dataset')
    X_train=pd.read_csv(config['data_split']['X_trainset_path'])
    y_train=pd.read_csv(config['data_split']['y_trainset_path'])
    X_test = pd.read_csv(config['data_split']['X_testset_path'])
    y_test = pd.read_csv(config['data_split']['y_testset_path'])
    y_train=np.ravel(y_train)
    y_test=np.ravel(y_test)
    logger.info('Train model')
    if estimator_name=='DL' or estimator_name=='DL_reg':
        model, history_df = DL_train(
            df_X_train=X_train,
            df_y_train=y_train,
            df_X_test=X_test,
            df_y_test=y_test,
            classifier_name=estimator_name,
            stoppings=config['train']['estimators'][estimator_name]['early_stoppings'],
            fit_hp=config['train']['estimators'][estimator_name]['fit_hp'],
        )
        logger.info("Best Validation Loss: {:0.4f}\nBest Validation Accuracy: {:1.4f}".format(history_df['val_loss'].min(), history_df['val_sparse_categorical_accuracy'].max()))
        logger.info('Save model')
        model_path = config['train']['estimators']['DL']['model_path']
        model.save(model_path,save_format='keras')                                                                                       

    else:
        model = train(
            df_X=X_train,
            df_y=y_train,
            estimator_name=estimator_name,
            param_grid=config['train']['estimators'][estimator_name]['param_grid'],
            cv=config['train']['cv']
        )
        logger.info(f'Best score: {model.best_score_}')
        logger.info(f'Best params: {model.best_params_}')
        logger.info('Save model')
        models_path = config['train']['model_path']
        joblib.dump(model, models_path)
    logger.info('Save metrics')
    # save best params
    reports_folder = Path(config['evaluate']['reports_dir'])
    metrics_path = reports_folder / config['train']['gridCV_file']
    try:
        json.dump(
        obj=model.best_params_,
        fp=open(metrics_path, 'w')
        )
    except:
        json.dump(
        obj={'model': 'DL'},
        fp=open(metrics_path, 'w')
        )



if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    train_model(config_path=args.config)