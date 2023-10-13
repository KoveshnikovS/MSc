import argparse
import joblib
import json
import pandas as pd
from pathlib import Path
from typing import Text
import yaml
from sklearn.metrics import f1_score, log_loss, ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from utils.logs import get_logger
from tensorflow import keras

def convert_to_labels(indexes, labels):
    result = []
    for i in indexes:
        l=labels[i]
        result.append(l)
    return result

def write_confusion_matrix_data(y_true, predicted, labels, filename):
    assert len(predicted) == len(y_true)
    predicted_labels = convert_to_labels(predicted, labels)
    true_labels = convert_to_labels(y_true, labels)
    cf = pd.DataFrame(list(zip(true_labels, predicted_labels)), columns=["y_true", "predicted"])
    cf.to_csv(filename, index=False)

def evaluate_model(config_path: Text) -> None:
    """### Evaluate model
    Assesses trained models both sklearn and TensorFlow on F1 and crossentropy score.
    Saves confusion matrix as .png and .csv files in /reports/ (can be changed in params.yaml)

    #### Parameters:
        - config_path {Text}: path to config, i.e. params.yaml
    
    #### Returns:
        - confusion_matrix.png
        - confusion_matrix.csv
        - metrics.json - metrics for single test (rewritten every test)
        - exp_reults.json - metrics for tests that are sequentially written (helps when doing a number of tests \
          - just copy them from file for e.g. visualization)

    """
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    logger = get_logger('EVALUATE', log_level=config['base']['log_level'])
    X_test = pd.read_csv(config['data_split']['X_testset_path'])
    y_test = pd.read_csv(config['data_split']['y_testset_path'])
    logger.info('Load model')

    if config['train']['estimator_name'] in ['DL','DL_reg']:
        model_path = config['train']['estimators']['DL']['model_path']
        model=keras.models.load_model(model_path)
        hyperparams=None
        prediction=model.predict(X_test)
        crossentropy=log_loss(y_true=y_test,y_pred=prediction)
        prediction=np.argmax(prediction,axis=1)
        f1=f1_score(y_true=y_test, y_pred=prediction, average='macro')
    else:
        model_path = config['train']['model_path']
        model = joblib.load(model_path)
        hyperparams=model.get_params()
        prediction = model.predict(X_test)
        f1 = f1_score(y_true=y_test, y_pred=prediction, average='macro')
        pred_proba = model.predict_proba(X_test)
        crossentropy=log_loss(y_true=y_test,y_pred=pred_proba)
    
    logger.info('Load test dataset')
    
    
    

    classes=['Healthy','1BRB','2BRB','3BRB','4BRB']
    options = [("Confusion matrix, {0}".format(config['train']['estimator_name']), None, 0),
               ("Normalized confusion matrix, {0}".format(config['train']['estimator_name']), 'true', 1)]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))  # Create subplots (1 row, 2 columns)
    plt.rc('font', size=14)  # Set fontsize
    for title, normalize, ax_idx in options:
        plt.grid(False)
        # main parameters of function `plot_confusion_matrix` are:
        # trained classifier (best_model), data (X_test, y_test)
        disp = ConfusionMatrixDisplay.from_predictions(y_test, prediction,
                                                     display_labels=classes,
                                                     cmap=plt.cm.Blues,
                                                     normalize=normalize, ax=axes[ax_idx])
        disp.ax_.set_title(title)

    cm=confusion_matrix(y_test,prediction)
    report = {
        'f1': f1,
        'crossentropy': crossentropy,
        'cm': cm,
        'actual': y_test,
        'predicted': prediction,
        'hyperparams': hyperparams
    }

    logger.info('Save metrics')
    # save metrics file
    reports_folder = Path(config['evaluate']['reports_dir'])
    metrics_path = reports_folder / config['evaluate']['metrics_file']
    try:
        json.dump(
            obj={'f1_score': report['f1'],
                 'crossentropy': report['crossentropy']
                 },
            fp=open(metrics_path, 'w')
        )
    except:
        json.dump(
            obj={'f1_score': report['f1'],
                 'crossentropy': report['crossentropy']
                 },
            fp=open(metrics_path, 'x')
        )
    #save to exp_results.json
    try:
        f=open(reports_folder / "exp_results.json",'r')
    except:
        json.dump(
            obj={'exp_name':[],
                'f1': [],
                 'crossentropy': []
                 },
            fp=open(reports_folder / "exp_results.json", 'x')
        )
    exp_results=json.load(f)
    f.close()
    exp_results["f1"].append(round(report['f1'],3))
    exp_results["crossentropy"].append(round(report['crossentropy'],3))
    exp_results["exp_name"].append(config["data_split"]["load_exclude"])
    json.dump(
        exp_results,
        fp=open(reports_folder / "exp_results.json", 'w')
    )
    logger.info(f'F1 metrics file saved to : {metrics_path}')

    logger.info('Save confusion matrix')
    # save confusion_matrix.png
    confusion_matrix_png_path = reports_folder / config['evaluate']['confusion_matrix_image']
    plt.savefig(confusion_matrix_png_path,dpi=300)
    logger.info(f'Confusion matrix saved to : {confusion_matrix_png_path}')

    confusion_matrix_data_path = reports_folder / config['evaluate']['confusion_matrix_data']
    write_confusion_matrix_data(np.ravel(y_test), np.ravel(prediction), labels=classes, filename=confusion_matrix_data_path)
    logger.info(f'Confusion matrix data saved to : {confusion_matrix_data_path}')


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    evaluate_model(config_path=args.config)                     