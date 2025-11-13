from typing import Iterable, Union
from collections import namedtuple
import multiprocessing
import os

import matplotlib.pyplot as plt
import numpy
import pandas
from scipy import spatial
from sklearn import decomposition as skd

from ccs import ccs
from cellohood.training.training import CelloPrediction


def divide_df_by_group(
    df, 
    group, 
    saving_path, 
    subdir_name='divide_df_group_by_cache', 
    filename_prefix='cache_df',
    generator=True,
):
    subdir_full_path = os.path.join(saving_path, subdir_name)
    if os.path.exists(subdir_full_path):
        raise ValueError(f'The cache directory: {subdir_full_path} exists.')
    os.makedirs(subdir_full_path)
    groups = pandas.unique(df[group])
    paths = []
    for group_nb, group_el in enumerate(groups):
        current_df = df[df[group] == group_el]
        current_df_filename = f'{group_nb}_{filename_prefix}_{group_el}.df'
        current_df_filepath = os.path.join(subdir_full_path, current_df_filename)
        current_df.to_csv(current_df_filepath)
        paths.append(current_df_filepath)
        if generator:
            yield current_df_filepath
    return paths

def get_apply_smooth_fun(
    df, 
    graph_result, 
    columns,
    index_col_name='s_nb',
):
    def _res(row):
        #raise ValueError(f'{row, index_col_name, row[index_col_name], columns, df.columns}')
        return df.loc[ # check
            graph_result[int(row[index_col_name])]
        ][columns].mean()
    return _res


def spatially_smooth_df(
    df_path,
    saving_path,
    columns_to_smooth,
    subdir_name='smoothed_subdir',
    spatial_colnames=('Centroid_X', 'Centroid_Y'),
    filename_prefix='smooth_cache_df',
    radius=100,
    index_col_name='s_nb',
    smoothing_postfix='s'
):
    result_columns = [
        f'{col}{smoothing_postfix}'
        for col in columns_to_smooth
    ]
    df_name = df_path.split('/')[-1]
    
    subdir_full_path = os.path.join(saving_path, subdir_name)
    os.makedirs(subdir_full_path, exist_ok=True)
        
    df = pandas.read_csv(df_path)
    df[index_col_name] = list(range(df.shape[0]))
    graph = spatial.cKDTree(df[list(spatial_colnames)])
    graph_result = graph.query_ball_point(
        df[list(spatial_colnames)], r=radius)
    means = df.apply(get_apply_smooth_fun(
        df, 
        graph_result, 
        index_col_name=index_col_name,
        columns=columns_to_smooth,
    ), axis=1)
    df[result_columns] = means
    current_df_filename = f'{filename_prefix}_{df_name}'
    saving_path = os.path.join(subdir_full_path, current_df_filename)
    df.to_csv(saving_path)
    return saving_path
    

def spatiall_smooth_df_from_tuple(
    tuple_,
):
    return spatially_smooth_df(*tuple_)


def spatiall_smooth_df_from_dict(
    dict_,
):
    return spatially_smooth_df(**dict_)


SmoothedDfResult = namedtuple(
    'SmoothedDfResult',
    [
        'result_df',
        'group',
        'radius',
        'spatial_colnames',
        'smoothed_col_names',
    ]
)


def smooth_df_multiprocessing(
    df, 
    group, 
    saving_path,
    pool_size=20,
    spatial_colnames=('Pos_X', 'Pos_Y'),
    radius=100,
    columns_to_smooth=None,
    smoothing_postfix='s'
):      
    if os.path.exists(saving_path):
        raise ValueError(f'Cache dir {saving_path} exists.')
    with multiprocessing.Pool(pool_size) as p:
        saved_paths = p.map(
            spatiall_smooth_df_from_dict,
            (
                {'df_path': df_path, 
                 'saving_path': saving_path,
                 'spatial_colnames': spatial_colnames,
                 'columns_to_smooth': columns_to_smooth,
                 'radius': radius,
                 'smoothing_postfix': smoothing_postfix,
                }
                for df_path in divide_df_by_group(df, group, saving_path)
            )
        )
    smoothed_col_names = [
        f'{col}{smoothing_postfix}'
        for col in columns_to_smooth
    ]
    result_df = pandas.concat(
        [
            pandas.read_csv(saved_path)
            for saved_path in saved_paths
        ]
    )
    return SmoothedDfResult(
        result_df=result_df,
        group=group,
        radius=radius,
        spatial_colnames=spatial_colnames,
        smoothed_col_names=smoothed_col_names,
    )

def get_table_index_from_filepath(filepath , cache_name: str = 'smooth_cache_df'):
    return int(filepath.split('/')[-1][len(cache_name) + 1:].split('_')[0])

    
def smooth_cellohood_prediction(
    cello_prediction: CelloPrediction,
    distance_threshold: float,
    cache_path: str,
):
    cell_cache_path = os.path.join(cache_path, 'CELL')
    smoothed_cell_predictions = smooth_df_multiprocessing(
        df=cello_prediction.cell_predictions,
        group=cello_prediction.image_id_column,
        saving_path=cell_cache_path,
        radius=distance_threshold,
        columns_to_smooth=cello_prediction.cello_columns,
        spatial_colnames=cello_prediction.pos_col_names,
    )
    bag_cache_path = os.path.join(cache_path, 'BAG')
    smoothed_bag_predictions = smooth_df_multiprocessing(
        df=cello_prediction.bag_predictions,
        group=cello_prediction.image_id_column,
        saving_path=bag_cache_path,
        radius=distance_threshold,
        spatial_colnames=cello_prediction.pos_col_names,
        columns_to_smooth=cello_prediction.cello_columns,
    )
    return CelloPrediction(
        cell_predictions=smoothed_cell_predictions.result_df,
        bag_predictions=smoothed_bag_predictions.result_df,
        cello_columns=smoothed_cell_predictions.smoothed_col_names,
        cellohood_neighborhood_cluster_colname=cello_prediction.cellohood_neighborhood_cluster_colname,
        image_id_column=cello_prediction.image_id_column,
        pos_col_names=cello_prediction.pos_col_names,
    )

    
LatentDirectionResult = namedtuple(
    'LatentDirectionResult',
    [
        'explained_variance_level_to_n_components',
        'pca_sm_data',
        'explained_variance_level_to_ccs_result',
        'pca',
    ]
)


def get_latent_directions(
    df,
    columns,
    explained_variance_levels: Iterable[Union[int, float]],
    min_nb_of_clusters: int,
    max_nb_of_clusters: int,
    clusters_step: int,
    nb_of_tries_per_cluster_nb: int,
    take_every_example_for_training: int,
):
    smoothed_data = df[columns]
    smd_pca = skd.PCA()
    pca_sm_data = smd_pca.fit_transform(smoothed_data)

    explained_variance_level_to_n_components = {}
    for explained_variance_level in explained_variance_levels:
        if isinstance(explained_variance_level, float):
            explained_cum_variance = smd_pca.explained_variance_ratio_.cumsum() 
            explained_variance_level_to_n_components[float(explained_variance_level)] = min(numpy.argmin(numpy.abs(explained_cum_variance - explained_variance_level)) + 1, len(columns))
        if isinstance(explained_variance_level, int):
            explained_variance_level_to_n_components[explained_variance_level] = explained_variance_level

    plt.plot(smd_pca.explained_variance_ratio_)
    plt.plot(smd_pca.explained_variance_ratio_.cumsum())
    for evl, n_comps in explained_variance_level_to_n_components.items():
        plt.vlines(
            n_comps, 
            ymin=0.0, ymax=1., 
            linestyles='--', 
            label=f'{round(100 * evl)}% variance: {n_comps}'
        )
    plt.legend()
    plt.show()

    explained_variance_level_to_ccs_result = {}
    for evl, n_comps in explained_variance_level_to_n_components.items():
        print(evl)
        current_data = pca_sm_data[:, :n_comps] 
        explained_variance_level_to_ccs_result[evl] = ccs.cellohood_cluster_selection(
            clusters_to_analyze=range(min_nb_of_clusters, max_nb_of_clusters, clusters_step),
            nb_of_tries_per_cluster_nb=nb_of_tries_per_cluster_nb, 
            take_every_example_for_training=take_every_example_for_training,
            x=current_data,
        )
    return LatentDirectionResult(
        explained_variance_level_to_n_components=explained_variance_level_to_n_components,
        explained_variance_level_to_ccs_result=explained_variance_level_to_ccs_result,
        pca_sm_data=pca_sm_data,
        pca=smd_pca,
    )


def summarize_latent_direction_result(latent_direction_result: LatentDirectionResult):
    print('Latent Direction Summary:')
    for explained_variance, dims in latent_direction_result.explained_variance_level_to_n_components.items():
        summary = f'Variance: {explained_variance} (dim={dims}), '
        summary += f'selected clusters = {latent_direction_result.explained_variance_level_to_ccs_result[explained_variance].best_clusterings}.'
        print(summary)

        
def get_latent_directions_predictions(
    latent_direction_result: LatentDirectionResult,
    explained_variance: float,
    selected_number_of_clusters: int,
):
    return ccs.batch_predict(
        latent_direction_result.explained_variance_level_to_ccs_result[explained_variance].best_clusterers[selected_number_of_clusters],
        latent_direction_result.pca_sm_data[:, :latent_direction_result.explained_variance_level_to_n_components[explained_variance]]
    )
