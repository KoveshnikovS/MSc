# import libs
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Text
import yaml
from utils.logs import get_logger
import argparse

# load deleting method from training set
def del_load(config_path: Text) -> None:
    """### Delete load level from training dataset and put it to test dataset
    From .csv file with Loading column, the method deletes the required load levels and places them to test set.

    #### Parameters:
    Parameters are taken directly from the params.yaml file.
    - config_path {Text}: path to config file, i.e. params.yaml

    #### Returns:
    Saves files to Data/processed/ (can be changed in params.yaml).
    - X_test.csv
    - X_train.csv
    - y_test.csv
    - y_train.csv
    """
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    logger = get_logger('DATA', log_level=config['base']['log_level'])

    logger.info('Get dataset')

    X_wo_load=pd.read_csv(config['data']['features'])

    X_train_path=config['data_split']['X_trainset_path']
    y_train_path=config['data_split']['y_trainset_path']
    load_train_path=config['data_split']['load_train_path']
    load_test_path=config['data_split']['load_test_path']
    X_test_path=config['data_split']['X_testset_path']
    y_test_path=config['data_split']['y_testset_path']

    if len(config['data_split']['load_exclude'])==0:
        X=X_wo_load.drop(columns=['Label']).copy()
        y=X_wo_load['Label'].copy()
        X_train, X_test,y_train, y_test = train_test_split(X,y,
                                                    test_size=config['data_split']['test_size'],
                                                    random_state=config['base']['random_state'])
        logger.info('Save train and test sets')
        
        load_train=X_train['Loading'].copy()
        load_test=X_test['Loading'].copy()
        X_train.drop(columns=['Loading'],inplace=True)
        X_test.drop(columns=['Loading'],inplace=True)
        X_train.to_csv(X_train_path,index=False)
        y_train.to_csv(y_train_path,index=False)
        X_test.to_csv(X_test_path,index=False)
        y_test.to_csv(y_test_path,index=False)
        load_train.to_csv(load_train_path,index=False)
        load_test.to_csv(load_test_path,index=False)
        logger.info('Saved')
    else:
        logger.info(X_wo_load.columns)
        X_load=pd.DataFrame()
        for level in config['data_split']['load_exclude']:
            #load level excluded from data in X_wo_load, X_load consists of only this load
            X_load=pd.concat([X_load, X_wo_load[X_wo_load['Loading']==level]],axis=0)
            X_wo_load.drop(index=X_wo_load[X_wo_load['Loading']==level].index,axis=0,inplace=True)

        # label sets created with the same logic, load and label columns dropped in X
        y_wo_load=X_wo_load['Label']
        y_load=X_load['Label']
        X_wo_load=X_wo_load.drop(columns=['Label']).copy()

        # X_wo_load=X_wo_load.drop(columns=['Loading']).copy()
        # X_load=X_load.drop(columns=['Loading']).copy()
        X_load=X_load.drop(columns=['Label']).copy()

        logger.info('Split features into train and test sets')
        #train/test split, for test data, excluded level added to it
        X_train_wo_load, X_test_wo_load,y_train_wo_load, y_test_wo_load = train_test_split(X_wo_load,
                                                                                            y_wo_load,
                                                                                            test_size=config['data_split']['test_size'],
                                                                                            random_state=config['base']['random_state'])
        X_test_wo_load=pd.concat([X_test_wo_load,X_load],axis=0)
        y_test_wo_load=pd.concat([y_test_wo_load,y_load],axis=0)


        load_train=X_train_wo_load['Loading'].copy()
        load_test=X_test_wo_load['Loading'].copy()
        X_train_wo_load.drop(columns=['Loading'],inplace=True)
        X_test_wo_load.drop(columns=['Loading'],inplace=True)
        #save
        logger.info('Save train and test sets')
        X_train_wo_load.to_csv(X_train_path,index=False)
        y_train_wo_load.to_csv(y_train_path,index=False)
        X_test_wo_load.to_csv(X_test_path,index=False)
        y_test_wo_load.to_csv(y_test_path,index=False)
        load_train.to_csv(load_train_path,index=False)
        load_test.to_csv(load_test_path,index=False)
        logger.info('Saved')


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    del_load(config_path=args.config)