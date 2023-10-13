import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew, median_abs_deviation, iqr
from scipy.fft import fft
from scipy.stats.mstats import idealfourths

def featurize(data: np.ndarray, window:np.ndarray, state: int, loading: float) -> pd.DataFrame:
    """ ## Calculate features
    This method calulates features from the passed window of current signal.\n 
    ### Parameters:

    - data -- time domain data
    - window -- windowing array, i.e. windowing function win = window(signal_length)
    - state -- num of BRB
    - loading -- loading level
    ### Return: 
    - pd.DataFrame with calculated features
    """
    fft_data=fft(np.ravel(data)*window)
    fft_data=np.abs(fft_data)
    #features calculated, can be easily extended, adding new line in the dictionary
    d={
        'T_mean': [np.mean(data)],
        'T_median': [np.median(data)],
        'T_kurtosis':kurtosis(data),
        'T_skewness':skew(data),
        'F_med_abs_dev':[median_abs_deviation(fft_data)],
        'F_median': [np.median(fft_data)],
        'F_kurtosis':[kurtosis(fft_data)],
        'F_skewness':[skew(fft_data)],
        'T_std': np.std(data),
        'T_med_abs_dev':median_abs_deviation(data),
        'T_mean_abs_dev':median_abs_deviation(data,center=np.mean),
        'T_sum_abs': np.abs(data).sum(),
        'T_max_abs': np.abs(data).max(),
        'T_sqrt_sum_abs_sqr': np.sqrt(np.power(np.abs(data),2).sum()),
        'F_mean': np.mean(fft_data),
        'F_std': np.std(fft_data),
        'F_mean_abs_dev':median_abs_deviation(fft_data,center=np.mean),
        'F_sum_abs':sum(np.abs(fft_data)),
        'F_max_abs':max(np.abs(fft_data)),
        'F_sqrt_sum_abs_sqr': np.sqrt(np.power(np.abs(fft_data),2).sum()),
        'Label': state,
        'Loading': loading
    }
    #features added to df
    features_df=pd.DataFrame(data=d)
    return features_df