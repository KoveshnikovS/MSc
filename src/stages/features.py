import argparse
import pandas as pd
from typing import Text, Dict
import yaml
import numpy as np
from utils.logs import get_logger
from utils.load_data import load_data
from utils.featurize import featurize
from scipy.signal.windows import flattop

class UnsupportedWindowing(Exception):

    def __init__(self, windowing_func):

        self.msg = f'Unsupported window function {windowing_func}'
        super().__init__(self.msg)

def get_supported_window() -> Dict:
    """
    Returns:
        Dict: supported classifiers
    """

    return {
        'blackman': np.blackman,
        'hanning': np.hanning,
        'hamming': np.hamming,
        'None': None,
        'flattop': flattop
    }

def featuring(config_path: Text) -> None:
    """### Create features from raw data
    Does windowing through the current signal for all experiments from .mat files. It envokes featurize.py for each current recording for feature creation.
    load_data.py loads the raw data for the processing.

    #### Parameters:
    - config_path {Text}: path to config
    #### Returns:
    - dataset.csv {file}: to Data/raw/ (can be changed in params.yaml). The dataset containes of features plus Label and Loading columns.
    """

    with open(config_path) as conf_file:
            config = yaml.safe_load(conf_file)

    logger = get_logger('FEATURES', log_level=config['base']['log_level'])

    #load raw data
    logger.info('Load raw data')
    exp_data=load_data()
    logger.info('Raw data loaded ')

    #get windowing function
    logger.info('Get windowig function')
    windowing = get_supported_window()
    windowing_func_name=config['features']['fft_window_func']

    if windowing_func_name not in windowing.keys():
        raise UnsupportedWindowing(windowing_func_name)

    logger.info('Windowig function {0}'.format(windowing_func_name))
    logger.info('Create features')
    #fft
    dataset=pd.DataFrame()
    states=[0,1,2,3,4] #number of BRB
    load_levels=[0.125,.25,.375,.50,.625,.75,.875,1] #loading levels of motor
    F_s=config['base']['grid_frequency'] #grid frequency
    window_num=config['features']['fft_window_num']
    window_size=config['features']['fft_window_size']
    feature_names=config['features']['names']
    i=0
    for state in states:
        j=0
        for level in load_levels:
                for win in range(window_num):
                    data=exp_data[i][j][win/F_s:(win+window_size)/F_s].copy()
                    if windowing[windowing_func_name]==None:
                        window=1
                    else:
                        window=windowing[windowing_func_name](len(data))
                    features=featurize(data,window,feature_names,state,level)
                    dataset=pd.concat([dataset,features],axis=0)
                j+=1
        i+=1
    logger.info('Features created')
    logger.info('Dataset size with label and loading columns {0}'.format(np.shape(dataset)))
    dataset.to_csv('Data/raw/dataset.csv',index=False)
    logger.info('Dataset saved')

if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    featuring(config_path=args.config) 