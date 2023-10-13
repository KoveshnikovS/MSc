import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from typing import Dict, Text
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier

class UnsupportedClassifier(Exception):

    def __init__(self, estimator_name):

        self.msg = f'Unsupported estimator {estimator_name}'
        super().__init__(self.msg)

def get_supported_estimator() -> Dict:
    """
    Returns:
        Dict: supported classifiers
    """

    return {
        'SVC': SVC,
        'GBC': GradientBoostingClassifier
    }

def train(df_X: pd.DataFrame,df_y:pd.DataFrame,
          estimator_name: Text, param_grid: Dict,  cv: int):
    """### Train SVC or GBC model
    Trains sklearn models SVC or GBC and returns the trained model.
    Training pipeline includes StandardScaler. GridSearchCV is used for hyperparameters tuning.
    #### Parameters:
        - df {pandas.DataFrame}: dataset
        - estimator_name {Text}: estimator name
        - param_grid {Dict}: grid parameters
        - cv {int}: cross-validation value of folds
    #### Returns:
        - trained model
    """
    estimators = get_supported_estimator()

    if estimator_name not in estimators.keys():
        raise UnsupportedClassifier(estimator_name)
    estimator = estimators[estimator_name]()

    pipe=Pipeline(steps=[('scaler', StandardScaler()),
                             (estimator_name,estimator)])
    clf=GridSearchCV(estimator=pipe,
                     param_grid=param_grid,
                     cv=cv,
                     scoring='neg_log_loss',
                     return_train_score=True,
                     verbose=1,
                     n_jobs=4)
    
    clf.fit(df_X,df_y)
    return clf