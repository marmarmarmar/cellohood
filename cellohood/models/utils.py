import numpy as np
import tensorflow as tf

from cellohood.data_processing import data_transforms


def base_variance_activation(x):
    return 0.1 + tf.math.softplus(x)


def minimization_loss(y_true, y_pred):
    return y_pred

    
def aggregate_preds_with_size_information(preds, graphs):
    is_sensible = graphs.sum(axis=-1, keepdims=True) > 1
    normalizing_factor = np.maximum(is_sensible.sum(axis=(1, 2)), 1.0).reshape((-1, 1))
    return (preds * is_sensible).sum(axis=1) / normalizing_factor

    
def get_cell_level_numpy_preds(
    full_model_preds,
    sizes,
):
    flattened_cell_preds = full_model_preds.reshape((-1, full_model_preds.shape[-1]))
    max_neighborhood_size = full_model_preds.shape[1]
    valid_cells_selector = np.array([
        i < size
        for size in sizes
        for i in range(max_neighborhood_size)
    ])
    return flattened_cell_preds[valid_cells_selector]


def get_cell_preds(preds, graphs):
    is_sensible = graphs.sum(axis=-1, keepdims=True) > 1
    normalizing_factor = np.maximum(is_sensible.sum(axis=(1, 2)), 1.0)
    return get_cell_level_numpy_preds(preds, normalizing_factor) 

    
def batch_predict(model, data, batch_size=1024):
    current_start = 0
    current_predictions = []
    while current_start < len(data):
        current_end = min(current_start + batch_size, len(data))
        current_predictions.append(model.predict(data[current_start:current_end]))
        current_start += batch_size
    return np.concatenate(current_predictions)


def get_marker_matrix_and_graph_array_from_df(
        df_,
        cellohood_cluster_colname,
        image_col_name,
        scaler,
        data_columns_col,
        max_neighborhood_size=None,
):
    sliced_df = slice_df_according_to_cellohood_cluster(
        df_=df_,
        cellohood_cluster_colname=cellohood_cluster_colname,
        image_col_name=image_col_name,
    )

    if max_neighborhood_size is None:
        max_neighborhood_size = max(
            [len(x_) for y_ in sliced_df
             for x_ in y_]
        )

    slides_np_data_arrays = [
       [
           neighborhood_df[data_columns_col].to_numpy()
           for neighborhood_df in slide_neighborhoods
       ]
       for slide_neighborhoods in sliced_df 
    ]

    padded_data_arrays = [
        [
            data_transforms.pad_np_array_on_first_dimension(
            np_array=np_data_array,
            pad_to=max_neighborhood_size,
            pad_with=0.0,
            )
            for np_data_array in slide_np_data_arrays
        ]
        for slide_np_data_arrays in slides_np_data_arrays
    ]

    scaled_and_transformed_data_arrays = [
        [
            scaler.transform(np_data_array)
            for np_data_array in slide_np_data_arrays
        ]
        for slide_np_data_arrays in padded_data_arrays
    ]

    unique_full_np_array = np.stack(
        [
            np_data_array
            for slide_np_data_arrays in scaled_and_transformed_data_arrays
            for np_data_array in slide_np_data_arrays
        ]
    )

    graph_matrices_ = [
        data_transforms.get_set_attention_map_padded_to_max_neighborhood(
            set_size=nb_df.shape[0],
            max_set_size=max_neighborhood_size,
            )
        for slide_neighborhoods in sliced_df
        for nb_df in slide_neighborhoods
    ]

    full_graph_matrix = np.stack(graph_matrices_)
    return unique_full_np_array, full_graph_matrix, max_neighborhood_size
    

def slice_df_according_to_cellohood_cluster(
        df_,
        cellohood_cluster_colname,
        image_col_name,
):
    slides_neighborhoods_dfs_ = []
    for image_name_ in df_[image_col_name].unique():
        current_image_array = df_[df_[image_col_name] == image_name_]
        current_neighborhood_dfs = []
        for neighborhood_index in current_image_array[cellohood_cluster_colname].unique():
            current_neighborhood_dfs.append(
                current_image_array[current_image_array[cellohood_cluster_colname] == neighborhood_index]
            )
        slides_neighborhoods_dfs_.append(current_neighborhood_dfs)
    return slides_neighborhoods_dfs_
