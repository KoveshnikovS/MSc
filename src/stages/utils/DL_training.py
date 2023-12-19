import pandas as pd
from typing import Dict, Text
from tensorflow import keras
from tensorflow.keras import layers 
from tensorflow.keras import regularizers

class UnsupportedClassifier(Exception):

    def __init__(self, classifier_name):

        self.msg = f'Unsupported classifier {classifier_name}'
        super().__init__(self.msg)

def get_supported_classifier() -> Dict:
    """
    Returns:
        Dict: supported classifiers
    """

    return {
        'DL': keras.Sequential, #MLP model w/o regularization
        'DL_reg':keras.Sequential, #MLP with regularization (only L2)
    }

def DL_train(df_X_train: pd.DataFrame,df_y_train:pd.DataFrame,
          df_X_test: pd.DataFrame, df_y_test:pd.DataFrame,
          classifier_name: Text, stoppings: Dict, fit_hp: Dict):
    """ ### Train DL model
    Trains MLP with TensorFlow library: DL and DL_reg. The architecture of the models is the same, the difference is that DL_reg has regularization.
    #### Parameters:
        - df {pandas.DataFrame}: dataset
        - classifier_name {Text}: estimator name
        - stoppings and fit_hp {Dict}: parameters
    #### Returns:
        trained model, history
    """

    classifiers = get_supported_classifier()
    if classifier_name not in classifiers.keys():
        raise UnsupportedClassifier(classifier_name)
    
    if classifier_name=='DL':
        classifier = classifiers[classifier_name]([
            layers.BatchNormalization(input_shape=[len(df_X_train.columns)]),
            layers.Dense(units=128,activation='swish'),
            layers.Dense(units=128,activation='selu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(units=128,activation='swish'),
            layers.Dense(units=4,activation='softmax'),
        ])

    elif classifier_name=='DL_reg':
        classifier=classifiers[classifier_name]([
            layers.BatchNormalization(input_shape=[len(df_X_train.columns)]),
            layers.Dense(units=128,activation='swish',kernel_regularizer=regularizers.L2(.01)),
            layers.Dense(units=128,activation='selu',kernel_regularizer=regularizers.L2(.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(units=128,activation='swish',kernel_regularizer=regularizers.L2(.001)),
            layers.Dense(units=5,activation='softmax'),
        ])
    classifier.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss='SparseCategoricalCrossentropy',
    metrics=['SparseCategoricalAccuracy'],
             jit_compile=True)

    early_stoppings=keras.callbacks.EarlyStopping(min_delta=stoppings['min_delta'],
                                                  patience=stoppings['patience'],
                                                  restore_best_weights=stoppings['restore_best_weights'],
                                                  start_from_epoch=stoppings['start_epoch'])
    history=classifier.fit(df_X_train,df_y_train,
                  validation_split=0.2,
                  batch_size=fit_hp['batch_size'],
                  epochs=fit_hp['epochs'],
                  callbacks=[early_stoppings],
                  verbose=1,
                  use_multiprocessing=True
                  )
    

    return classifier, pd.DataFrame(history.history)
