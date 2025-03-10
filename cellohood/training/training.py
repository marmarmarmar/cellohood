from collections import namedtuple
from typing import Any 
from typing import List 
from typing import Optional
from typing import Union 
import json
import os

from keras import optimizers as k_optimizers
from keras import callbacks as k_callbacks
from sklearn.decomposition import PCA 
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import pandas
import sklearn.preprocessing as skp

from cellohood.models import autoencoders as ca
from cellohood.models import utils as imu
from cellohood.models import saving_utils as isu
from cellohood.data_processing import data_transforms as cdt


def split_bag_cello_df_by(
    cello_df,
    by: str,
    split_percentage: float,
    random_seed: int = 42,
):
    patients = list(set(cello_df.cell_df[by]))
    if split_percentage < 1.0:
        train_patients, test_patients = train_test_split(
            patients,
            train_size=split_percentage,
            random_state=random_seed,
        )
    else:
        train_patients, test_patients = patients, []

    train_cell_df = cello_df.cell_df[cello_df.cell_df[by].apply(lambda x: x in train_patients)]
    if split_percentage < 1.0:
        test_cell_df = cello_df.cell_df[cello_df.cell_df[by].apply(lambda x: x in test_patients)]

    train_bag_cello_df =  cdt.BagCelloDf(
        cell_df=train_cell_df,
        image_id_column=cello_df.image_id_column,
        distance_threshold=cello_df.distance_threshold,
        marker_col_names=cello_df.marker_col_names,
        pos_col_names=cello_df.pos_col_names,
        cellohood_neighborhood_cluster_colname=cello_df.cellohood_neighborhood_cluster_colname,
    )

    if split_percentage < 1.0:
        test_bag_cello_df =  cdt.BagCelloDf(
            cell_df=test_cell_df,
            image_id_column=cello_df.image_id_column,
            distance_threshold=cello_df.distance_threshold,
            marker_col_names=cello_df.marker_col_names,
            pos_col_names=cello_df.pos_col_names,
            cellohood_neighborhood_cluster_colname=cello_df.cellohood_neighborhood_cluster_colname,
        )
        return train_bag_cello_df, test_bag_cello_df
    return train_bag_cello_df, None

    
StandardizedBagCelloDfs = namedtuple(
    'StandardizedBagCelloDfs',
    [
        'train_cello_df',
        'test_cello_df',
        'marker_scaler',
        'marker_col_names',
        'output_size',
    ]
)


def standardize_bag_cello_df(
    train_bag_cello_df,
    marker_col_names: List[str],
    test_bag_cello_df=None,
    marker_scaler: Union[str, Any] = 'standard',
    pca_dimensions: int = 0,
    inplace: bool = False,
    fit_marker_scaler: bool = True
):

    if marker_scaler == 'standard':
        marker_scaler = skp.StandardScaler()
    if marker_scaler == 'min-max':
        marker_scaler = skp.MinMaxScaler()
    if marker_scaler is None:
        marker_scaler = skp.FunctionTransformer(func=lambda x: x, inverse_func=lambda x: x)

    if not inplace:
        train_df = train_bag_cello_df.cell_df.copy()
        if test_bag_cello_df is not None:
            test_df = test_bag_cello_df.cell_df.copy()
    else:
        train_df = train_bag_cello_df.cell_df
        if test_bag_cello_df is not None:
            test_df = test_bag_cello_df.cell_df

    scaler_train_array = train_df[marker_col_names].to_numpy()
    if pca_dimensions > 0 and fit_marker_scaler:
        pca_ = PCA(pca_dimensions)
        marker_scaler = Pipeline([('scaler', marker_scaler), ('pca', pca_)])
    if fit_marker_scaler:
        marker_scaler.fit(scaler_train_array)
    train_df[marker_col_names] = marker_scaler.transform(scaler_train_array)
    if test_bag_cello_df is not None:
        test_df[marker_col_names] = marker_scaler.transform(test_df[marker_col_names].values) 
    if inplace:
        return StandardizedBagCelloDfs(
            train_cello_df=train_bag_cello_df,
            test_cello_df=test_bag_cello_df,
            marker_scaler=marker_scaler,
            marker_col_names=marker_col_names,
            output_size=len(marker_col_names) if pca_dimensions == 0 else pca_dimensions
        )
    
    train_bag_cello_df =  cdt.BagCelloDf(
        cell_df=train_df,
        image_id_column=train_bag_cello_df.image_id_column,
        distance_threshold=train_bag_cello_df.distance_threshold,
        marker_col_names=train_bag_cello_df.marker_col_names,
        pos_col_names=train_bag_cello_df.pos_col_names,
        cellohood_neighborhood_cluster_colname=train_bag_cello_df.cellohood_neighborhood_cluster_colname,
    )
    if test_bag_cello_df is not None:
        test_bag_cello_df =  cdt.BagCelloDf(
            cell_df=test_df,
            image_id_column=test_bag_cello_df.image_id_column,
            distance_threshold=test_bag_cello_df.distance_threshold,
            marker_col_names=test_bag_cello_df.marker_col_names,
            pos_col_names=test_bag_cello_df.pos_col_names,
            cellohood_neighborhood_cluster_colname=test_bag_cello_df.cellohood_neighborhood_cluster_colname,
        )
    return StandardizedBagCelloDfs(
        train_cello_df=train_bag_cello_df,
        test_cello_df=test_bag_cello_df,
        marker_scaler=marker_scaler,
        marker_col_names=marker_col_names,
        output_size=len(marker_col_names) if pca_dimensions == 0 else pca_dimensions
    )


    
CellohoodModel = namedtuple(
    'CelloTrainResults',
    [
        'model',
        'history',
        'marker_scaler',
        'marker_col_names',
        'output_size',
        'max_neighborhood_size',
    ]
)


def train(
        standardized_cello_dfs : StandardizedBagCelloDfs,
        batch_size: int = 512,
        epoch_nb: int = 1000,
        optimizer_lr: float = 0.0001,
        optimizer_clipnorm: float = 1.0,
        use_cuda: bool = True,
        layer_size: int = 128,
        intermediate_layer_size: int = 256,
        latent_size: int = 64,
        tensorboard_path: Optional[str] = None,
):

    train_array, train_graph_array, max_neighborhood_size = imu.get_marker_matrix_and_graph_array_from_df(
        df_=standardized_cello_dfs.train_cello_df.cell_df,
        cellohood_cluster_colname=standardized_cello_dfs.train_cello_df.cellohood_neighborhood_cluster_colname,
        image_col_name=standardized_cello_dfs.train_cello_df.image_id_column,
        scaler=skp.FunctionTransformer(func=lambda x: x, inverse_func=lambda x: x),
        data_columns_col=standardized_cello_dfs.marker_col_names,
        max_neighborhood_size=None,
    )

    test_array, test_graph_array = None, None
    if standardized_cello_dfs.test_cello_df is not None:
        test_array, test_graph_array, _ = imu.get_marker_matrix_and_graph_array_from_df(
            df_=standardized_cello_dfs.test_cello_df.cell_df,
            cellohood_cluster_colname=standardized_cello_dfs.train_cello_df.cellohood_neighborhood_cluster_colname,
            image_col_name=standardized_cello_dfs.train_cello_df.image_id_column,
            scaler=skp.FunctionTransformer(func=lambda x: x, inverse_func=lambda x: x),
            data_columns_col=standardized_cello_dfs.marker_col_names,
            max_neighborhood_size=max_neighborhood_size,
        )

    if not use_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        
    callbacks = []
    if tensorboard_path is not None:
        callbacks.append(k_callbacks.TensorBoard(log_dir=tensorboard_path))

    cellohood_model = ca.BaseWinterCellEnvironmentAEV2(
        max_neighborhood_size=max_neighborhood_size,
        output_size=standardized_cello_dfs.output_size,
        layer_size=layer_size,
        intermediate_layer_size=intermediate_layer_size,
        latent_size=latent_size,
    )
    
    cellohood_model.compile(
        loss=imu.minimization_loss,
        optimizer=k_optimizers.adam_v2.Adam(learning_rate=optimizer_lr, global_clipnorm=optimizer_clipnorm),
    )

    history = cellohood_model.fit(
        x=[train_array, train_array, train_graph_array], 
        y=train_array,
        validation_data=[[test_array, test_array, test_graph_array], test_array] if test_array is not None else None,
        epochs=epoch_nb,
        batch_size=batch_size,
        verbose=2,
        callbacks=callbacks,
    )

    return CellohoodModel(
        model=cellohood_model,
        history=history,
        marker_scaler=standardized_cello_dfs.marker_scaler,
        marker_col_names=standardized_cello_dfs.marker_col_names,
        output_size=latent_size,
        max_neighborhood_size=max_neighborhood_size,
    )

    
CelloPrediction = namedtuple(
    'CelloPrediction',
    [
        'cell_predictions',
        'bag_predictions',
        'cello_columns',
        'cellohood_neighborhood_cluster_colname',
        'image_id_column',
        'pos_col_names',
    ]
)

    
def run_prediction_on_cello_df(
    cellohood_training_result: CellohoodModel,
    bag_cello_df,
    marker_scaler=None,
    cello_columns=None,
):
    if marker_scaler is None:
        marker_scaler = cellohood_training_result.marker_scaler
    if marker_scaler == 'identity':
        marker_scaler = skp.FunctionTransformer(func=lambda x: x, inverse_func=lambda x: x)

    if cello_columns is None:
        cello_columns = [f'{i}c' for i in range(cellohood_training_result.output_size)]
        
    array, graph_array, _ = imu.get_marker_matrix_and_graph_array_from_df(
        df_=bag_cello_df.cell_df,
        cellohood_cluster_colname=bag_cello_df.cellohood_neighborhood_cluster_colname,
        image_col_name=bag_cello_df.image_id_column,
        scaler=marker_scaler,
        data_columns_col=cellohood_training_result.marker_col_names,
        max_neighborhood_size=cellohood_training_result.max_neighborhood_size,
    )

    full_cell_results = cellohood_training_result.model.encoder.predict([array, array, graph_array])
    cell_level_results = pandas.DataFrame(
        imu.get_cell_preds(full_cell_results, graph_array),
        columns=cello_columns,
    )
    cell_level_results[bag_cello_df.image_id_column] = bag_cello_df.cell_df[bag_cello_df.image_id_column]
    cell_level_results[bag_cello_df.pos_col_names] = bag_cello_df.cell_df[bag_cello_df.pos_col_names]
    bag_level_results = pandas.DataFrame(
        imu.aggregate_preds_with_size_information(full_cell_results, graph_array),
        columns=cello_columns,
    )
    cell_level_results[bag_cello_df.cellohood_neighborhood_cluster_colname] = bag_cello_df.cell_df[
        bag_cello_df.cellohood_neighborhood_cluster_colname
    ]
    bag_level_results[bag_cello_df.image_id_column] = bag_cello_df.cell_df.groupby(
        bag_cello_df.cellohood_neighborhood_cluster_colname)[bag_cello_df.image_id_column].first()
    bag_level_results[bag_cello_df.pos_col_names] = bag_cello_df.cell_df.groupby(
        bag_cello_df.cellohood_neighborhood_cluster_colname)[bag_cello_df.pos_col_names].mean()
    return CelloPrediction(
        cell_predictions=cell_level_results,
        bag_predictions=bag_level_results,
        cello_columns=cello_columns,
        cellohood_neighborhood_cluster_colname=bag_cello_df.cellohood_neighborhood_cluster_colname,
        image_id_column=bag_cello_df.image_id_column,
        pos_col_names=bag_cello_df.pos_col_names,
    ) 


def save_cello_train_result(
    cello_train_results,
    path: str,
):
    os.makedirs(path, exist_ok=True)
    isu.save_model(model=cello_train_results.model, path=path)
    isu.save_model_history(history=cello_train_results.history, path=path)
    isu.save_scaler(scaler=cello_train_results.marker_scaler, path=path)
    ctr_json = {
       'marker_col_names':  cello_train_results.marker_col_names,
       'output_size': cello_train_results.output_size,
       'max_neighborhood_size': cello_train_results.max_neighborhood_size,
    }
    ctr_json_path = os.path.join(path, isu.CTR_JSON_FILENAME)
    with open(ctr_json_path, 'w', encoding='utf-8') as f:
        json.dump(ctr_json, f, ensure_ascii=False, indent=4)


def load_cello_train_result(
    path: str,
):
    model = isu.load_model(path=path)
    history = isu.load_model_history(path=path)
    scaler = isu.load_scaler(path=path)
    ctr_json_path = os.path.join(path, isu.CTR_JSON_FILENAME)
    with open(ctr_json_path, 'r', encoding='utf-8') as f:
        ctr_json = json.load(f)
    return CellohoodModel(
        model=model,
        history=history,
        marker_scaler=scaler,
        marker_col_names=ctr_json['marker_col_names'],
        output_size=ctr_json['output_size'],
        max_neighborhood_size=ctr_json['max_neighborhood_size'],
    )
