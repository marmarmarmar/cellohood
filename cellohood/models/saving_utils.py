import os
import json
from typing import Any
from typing import List 

import joblib
import numpy as np
import pandas
from keras import callbacks as k_callbacks
from keras import models as k_models

from cellohood.models.autoencoders import BaseWinterCellEnvironmentAEV2
from cellohood.models import utils as imu


CELL_TYPES_HISTOGRAM_SUBDIR = 'cell_types_histogram'
CELL_TYPES_HISTOGRAM_DF_NAME = 'cell_types_histogram.csv'
CTR_JSON_FILENAME = 'ctr.json'
MEAN_MARKER_SUBDIR = 'mean_marker'
MODEL_CLASS_FILENAME = 'model_class.txt'
MODEL_FULL_PREDICTIONS_FILENAME = 'model_full_preds.npy'
MODEL_HISTORY_FILENAME = 'model_history.df'
MODEL_JSON_FILENAME = 'model.json'
MODEL_PREDICTIONS_FILENAME = 'model_preds.npy'
MODEL_SAVING_SUBDIR = 'model'
MODEL_SCALER_FILENAME = 'scaler.joblib'
MODEL_TRAIN_TEST_PATIENTS_FILENAME = 'train_test_patients.json'
MODEL_TRAINING_PARAMS_FILENAME = 'training_params.json'
MODEL_WEIGHTS_FILENAME = 'model_weights.h5'


def save_cell_type_histogram(path: str, cell_types_histogram_df: pandas.DataFrame):
    full_cell_type_histogram_path = os.path.join(path, CELL_TYPES_HISTOGRAM_DF_NAME)
    cell_types_histogram_df.to_csv(full_cell_type_histogram_path)


def save_cellohood_preds(path: str, preds: np.array, graphs: np.array):
    full_preds_full_path = os.path.join(path, MODEL_FULL_PREDICTIONS_FILENAME)
    preds_full_path = os.path.join(path, MODEL_PREDICTIONS_FILENAME)
    aggregated_preds = imu.aggregate_preds_with_size_information(preds=preds, graphs=graphs)
    np.save(full_preds_full_path, preds)
    np.save(preds_full_path, aggregated_preds)


def save_preds(path: str, preds: np.array):
    preds_full_path = os.path.join(path, MODEL_PREDICTIONS_FILENAME)
    np.save(preds_full_path, preds)


def save_scaler(path: str, scaler: Any):
    full_scaler_path = os.path.join(path, MODEL_SCALER_FILENAME)
    joblib.dump(value=scaler, filename=full_scaler_path)


def load_scaler(path: str):
    full_scaler_path = os.path.join(path, MODEL_SCALER_FILENAME)
    return joblib.load(filename=full_scaler_path)

    
def save_train_test_patients(path: str, train_patients: List[str], test_patients: List[str]):
    train_test_patients_path = os.path.join(path, MODEL_TRAIN_TEST_PATIENTS_FILENAME) 
    patients_json = {
        'train': train_patients,
        'test': test_patients,
    }
    with open(train_test_patients_path, 'w', encoding='utf-8') as f:
        json.dump(patients_json, f, ensure_ascii=False, indent=4)


def jsonify_keras_model_history(history: k_callbacks.History):
    return {
        'params': history.params,
    }


def save_model_history(path: str, history: k_callbacks.History):
    training_params_json_full_path = os.path.join(path, MODEL_TRAINING_PARAMS_FILENAME)
    model_history_full_path = os.path.join(path, MODEL_HISTORY_FILENAME)
    with open(training_params_json_full_path, 'w', encoding='utf-8') as f:
        json.dump(jsonify_keras_model_history(history=history), f, ensure_ascii=False, indent=4)
    pandas.DataFrame(history.history).to_csv(model_history_full_path)

    
def load_model_history(path: str):
    training_params_json_full_path = os.path.join(path, MODEL_TRAINING_PARAMS_FILENAME)
    model_history_full_path = os.path.join(path, MODEL_HISTORY_FILENAME)
    with open(training_params_json_full_path, 'r', encoding='utf-8') as f:
        history_params = json.load(f)
    history = pandas.read_csv(model_history_full_path)
    return {
        'params': history_params,
        **{
            col: list(history[col])
            for col in history       
        }
    }
    

def save_BaseWinterCellEnvironmentAEV2(
    model: k_models.Model,
    path: str,
    model_weights_filename: str = MODEL_WEIGHTS_FILENAME,
    model_json_filename: str = MODEL_JSON_FILENAME,
):
    if not isinstance(model, BaseWinterCellEnvironmentAEV2):
        raise ValueError(
            f'Function save_BaseWinterCellEnvironmentAEV2 might only save BaseWinterCellEnvironmentAEV2 model type.'
            f' Instead it was provided with {model} instance.'
        )
        
    os.makedirs(path, exist_ok=True)
    weights_full_path = os.path.join(path, model_weights_filename)
    model_json_full_path = os.path.join(path, model_json_filename)
    model.save_weights(weights_full_path)
    model_json = {
        'output_size': model.output_size,
        'max_neighborhood_size': model.max_neighborhood_size,
        'latent_size': model.latent_size,
    }
    with open(model_json_full_path, 'w', encoding='utf-8') as f:
        json.dump(model_json, f, ensure_ascii=False, indent=4)


def load_BaseWinterCellEnvironmentAEV2(
    path: str,
    model_weights_filename: str = MODEL_WEIGHTS_FILENAME,
    model_json_filename: str = MODEL_JSON_FILENAME,
):
    weights_full_path = os.path.join(path, model_weights_filename)
    model_json_full_path = os.path.join(path, model_json_filename)
    with open(model_json_full_path, 'r', encoding='utf-8') as f:
        model_json = json.load(f)
    model = BaseWinterCellEnvironmentAEV2(
        output_size=model_json['output_size'],
        max_neighborhood_size=model_json['max_neighborhood_size']
    ) 
    dummy_input = np.zeros((1, model_json['max_neighborhood_size'], model_json['output_size']))
    dummy_graph = np.zeros((1, model_json['max_neighborhood_size'], model_json['max_neighborhood_size']))
    model.predict([dummy_input, dummy_input, dummy_graph])
    model.load_weights(weights_full_path)
    return model

    
def save_model(
    model: k_models.Model,
    path: str,
    model_class_filename: str = MODEL_CLASS_FILENAME,
):
    model_class_type_str = model.__class__.__name__
    if model_class_type_str not in TYPE_TO_SAVING_FUNCTION:
        raise ValueError(f'Unknown model type: {model_class_type_str}.') 
    saving_function = TYPE_TO_SAVING_FUNCTION[model_class_type_str]
    model_class_filename_path = os.path.join(path, model_class_filename)
    with open(model_class_filename_path, 'w') as f:
        f.write(model_class_type_str)
    saving_function(model=model, path=path)


def load_model(
    path: str,
    model_class_filename: str = MODEL_CLASS_FILENAME,
):
    model_class_filename_path = os.path.join(path, model_class_filename)
    with open(model_class_filename_path, 'r') as f:
        model_class_type_str = f.readline()
    if model_class_type_str not in TYPE_TO_LOADING_FUNCTION:
        raise ValueError(f'Unknown model type: {model_class_type_str}.') 
    loading_function = TYPE_TO_LOADING_FUNCTION[model_class_type_str]
    return loading_function(path=path)
    

TYPE_TO_SAVING_FUNCTION = {
   'BaseWinterCellEnvironmentAEV2': save_BaseWinterCellEnvironmentAEV2, 
}


TYPE_TO_LOADING_FUNCTION = {
   'BaseWinterCellEnvironmentAEV2': load_BaseWinterCellEnvironmentAEV2, 
}
