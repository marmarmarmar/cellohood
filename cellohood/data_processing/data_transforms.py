from collections import defaultdict
from collections import namedtuple
from typing import List
from typing import Tuple

from sklearn import cluster as sk_cluster
import numpy as np
import pandas


ClusteredCellDfWithDistanceThreshold = namedtuple(
    'ClusteredCellDfWithDistanceThreshold',
    [
        'cluster_assignment',
        'cluster_to_cells_assignment'
    ]
)


BagCelloDf = namedtuple(
    'CelloDf',
    [
        'cell_df',
        'image_id_column',
        'pos_col_names',
        'distance_threshold',
        'cellohood_neighborhood_cluster_colname',
        'marker_col_names',
    ]
)


def cluster_cell_df_positions_within_distance_threshold(
        cell_df,
        position_columns,
        distance_threshold=25.0,
):
    """
    Cluster cells from the cell_df, w.r.t. to positions encoded by position_columns using
    sklearn.AgglomerativeClustering

    :param cell_df: Dataframe with cells.
    :param position_columns: Column names with cell positions.
    :param distance_threshold: Threshold on the distance between the cells.
    :return: ClusteredCellDfWithDistanceThreshold object with:
        cluster_assignment: list of len of nb of rows in cell_df with cluster index assigned.
        cluster_to_cells_assignment: a dictionary with keys being cluster indices and values: indices of cells assigned
        to a given cluster.
    """
    center_data = cell_df[position_columns].to_numpy()
    clusterer = sk_cluster.AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold)
    cluster_assignment = [int(c) for c in list(clusterer.fit_predict(center_data).astype('int'))]
    cluster_to_cells_assignment = defaultdict(list)
    for el_ind, cl_as in enumerate(cluster_assignment):
        cluster_to_cells_assignment[cl_as].append(el_ind)
    return ClusteredCellDfWithDistanceThreshold(
        cluster_assignment=cluster_assignment,
        cluster_to_cells_assignment=cluster_to_cells_assignment,
    )


def get_set_attention_map_for_clusterings(
        cluster_to_cells_assignment,
        max_neighborhood_size,
):
    return {
        cluster_index: get_set_attention_map_padded_to_max_neighborhood(
            set_size=len(cluster_to_cells_assignment[cluster_index]),
            max_set_size=max_neighborhood_size,
        )
        for cluster_index in cluster_to_cells_assignment
    }


def get_set_attention_map_padded_to_max_neighborhood(
        set_size: int,
        max_set_size: int,
):
    set_attention_map = np.eye(max_set_size)
    limit = min(set_size, max_set_size)
    for row_ind in range(limit):
        for col_ind in range(limit):
            set_attention_map[row_ind, col_ind] = 1
    return set_attention_map


def extract_neighborhoods_df_from_clusterings(
        cell_df,
        clustered_cells,
):
    return [
        cell_df.iloc[cluster_els_inds]
        for _, cluster_els_inds in clustered_cells.cluster_to_cells_assignment.items()
    ]


def enrich_position_col_name_with_neighborhood_postfix(
    position_col_name
):
    return f'{position_col_name}_nc'


def enrich_neighborhood_with_neighborhood_centered_positions(
        neighborhood_df,
        position_columns,
):
    for position_column in position_columns:
        neighborhood_df[
            enrich_position_col_name_with_neighborhood_postfix(position_column)
        ] = neighborhood_df[position_column] - neighborhood_df[position_column].mean()
    return neighborhood_df


def pad_np_array_on_first_dimension(
        np_array,
        pad_to,
        pad_with=0.0,
):
    if np_array.shape[0] >= pad_to:
        return np_array[:pad_to]
    difference = pad_to - np_array.shape[0]
    padding = np.ones(shape=(difference,) + np_array.shape[1:]) * pad_with
    return np.concatenate([np_array, padding], axis=0)


def arcsinhtransform(x):
    return np.arcsinh(x / 5.0)


def get_neighborhood_array_dot_product_matrix(
        neighborhood_array,
        position_columns,
        pad_neighborhood_to=None,
):
    neighborhood_position_array = neighborhood_array[position_columns].to_numpy()
    if pad_neighborhood_to is not None:
        if pad_neighborhood_to > neighborhood_position_array.shape[0]:
            padded_array = np.zeros(shape=(pad_neighborhood_to, neighborhood_position_array.shape[1]))
            padded_array[:neighborhood_position_array.shape[0]] = neighborhood_position_array
            return get_outer_dot_product_matrix(padded_array)
        return get_outer_dot_product_matrix(neighborhood_position_array[:pad_neighborhood_to])
    return get_outer_dot_product_matrix(neighborhood_position_array)


def get_outer_dot_product_matrix(
        np_array,
):
    neighborhood_size = np_array.shape[0]
    result_matrix = np.zeros(
        shape=(neighborhood_size, neighborhood_size),
    )
    for row_ind in range(neighborhood_size):
        for col_ind in range(neighborhood_size):
            l = np_array[row_ind]
            r = np_array[col_ind]
            result_matrix[row_ind, col_ind] = (l * r).sum()
    return result_matrix


def get_neighborhood_array_l2_distance_matrix(
        neighborhood_array,
        position_columns,
        pad_neighborhood_to=None,
):
    neighborhood_position_array = neighborhood_array[position_columns].to_numpy()
    if pad_neighborhood_to is not None:
        if pad_neighborhood_to > neighborhood_position_array.shape[0]:
            padded_array = np.zeros(shape=(pad_neighborhood_to, neighborhood_position_array.shape[1]))
            padded_array[:neighborhood_position_array.shape[0]] = neighborhood_position_array
            return get_outer_l2_product_matrix(padded_array)
        return get_outer_l2_product_matrix(neighborhood_position_array[:pad_neighborhood_to])
    return get_outer_l2_product_matrix(neighborhood_position_array)


def get_outer_l2_product_matrix(
        np_array,
):
    neighborhood_size = np_array.shape[0]
    result_matrix = np.zeros(
        shape=(neighborhood_size, neighborhood_size),
    )
    for row_ind in range(neighborhood_size):
        for col_ind in range(neighborhood_size):
            l = np_array[row_ind]
            r = np_array[col_ind]
            result_matrix[row_ind, col_ind] = ((l - r) ** 2).sum() ** 0.5
    return result_matrix


def get_outer_marker_product_of_neighborhood_df(
        neighborhood_df,
        data_columns,
        pad_to=None,
):
    neighborhood_size = neighborhood_df.shape[0]
    data_array = neighborhood_df[data_columns].to_numpy()
    if pad_to is not None:
        if pad_to > data_array.shape[0]:
            padded_array = np.zeros(shape=(pad_to, data_array.shape[1]))
            padded_array[:data_array.shape[0]] = data_array
            return get_outer_product_of_a_marker_array(padded_array)
        else:
            padded_array = data_array[:pad_to]
            return get_outer_product_of_a_marker_array(padded_array)
    return get_outer_product_of_a_marker_array(data_array)


def get_outer_product_of_a_marker_array(np_array):
    result_array = np.zeros(shape=(np_array.shape[0], np_array.shape[0], np_array.shape[1]))
    for l_index in range(result_array.shape[0]):
        for r_index in range(result_array.shape[0]):
            result_array[l_index, r_index] = 0.5 * (np_array[l_index] + np_array[r_index])
    return result_array


def get_snake_flattened_marker_array_of_neighborhood_df(
        neighborhood_df,
        data_columns,
):
    neighborhood_size = neighborhood_df.shape[0]
    data_array = neighborhood_df[data_columns].to_numpy()
    result_array = np.zeros(
        shape=(
            int(neighborhood_size * (neighborhood_size + 1) * 0.5),
            len(data_columns)
        )
    )
    current_index = 0
    for row_ind in range(neighborhood_size):
        for col_ind in range(row_ind + 1):
            result_array[current_index] = (data_array[row_ind] + data_array[col_ind]) * 0.5
            current_index += 1
    return result_array


def snake_flatten_symmetrical_matrix(
        sym_matrix,
):
    size_of_matrix = sym_matrix.shape[0]
    result_array = np.zeros(shape=(int(size_of_matrix * (size_of_matrix + 1) * 0.5),))
    current_count = 0
    for row_ind in range(size_of_matrix):
        for col_ind in range(row_ind + 1):
            result_array[current_count] = sym_matrix[row_ind, col_ind]
            current_count += 1
    return result_array


def get_snake_square(x_):
    return int(x_ * (x_ + 1) * 0.5)

    
def transform_df_to_bag_cello_df(
    df_,
    marker_col_names: List[str],
    image_id_column: str = 'RoiID',
    pos_col_names: Tuple[str] = ('Pos_X', 'Pos_Y'),
    distance_threshold: float = 25.,
    cellohood_neighborhood_cluster_colname: str = 'cellohood_neighborhood_cluster'
):
    full_slides_dfs = []
    current_rois_ids = set(df_[image_id_column])
    for roi_id in current_rois_ids:
            full_slides_dfs.append(df_[df_[image_id_column] == roi_id].copy())
    
    # Cluster cells on the slides
    slides_clusterings = [
        cluster_cell_df_positions_within_distance_threshold(
            cell_df=full_slide_df,
            position_columns=pos_col_names,
            distance_threshold=distance_threshold,
            
        )
        for full_slide_df in full_slides_dfs
    ]

    # Extend the full slides dfs with the tumorhood cluster index
    cummulative_counter = 0
    for full_slide_df, slide_clustering in zip(full_slides_dfs, slides_clusterings):
        inverse_clustering_assignment = {
            cluster_: ind_ + cummulative_counter for ind_, cluster_ in enumerate(slide_clustering.cluster_to_cells_assignment)
        }
        inversed_clusters = [
            inverse_clustering_assignment[cluster_ind] 
            for cluster_ind in slide_clustering.cluster_assignment
        ]
        full_slide_df[cellohood_neighborhood_cluster_colname] = inversed_clusters
        cummulative_counter += len(slide_clustering.cluster_to_cells_assignment)
        
    # Extract per slide neighborhoods
    slides_neighborhoods_dfs = [
        extract_neighborhoods_df_from_clusterings(
            cell_df=full_slide_df,
            clustered_cells=slide_clustering
        )
        for full_slide_df, slide_clustering in zip(full_slides_dfs, slides_clusterings)
    ]
    
    cello_df = pandas.concat(
        [
            neighobrhood_df
            for slide_neighborhoods in slides_neighborhoods_dfs
            for neighobrhood_df in slide_neighborhoods
        ],
        ignore_index=True,
    )

    return BagCelloDf(
        cell_df=cello_df,
        image_id_column=image_id_column,
        pos_col_names=pos_col_names,
        distance_threshold=distance_threshold,
        cellohood_neighborhood_cluster_colname=cellohood_neighborhood_cluster_colname,
        marker_col_names=marker_col_names,
    )


def apply_arcsinh_to_df(
    df_,
    marker_col_names: List[str],
    inplace: bool = False,
):
    if not inplace:
        df_ = df_.copy()
    df_[marker_col_names] = arcsinhtransform(df_[marker_col_names].values)
    return df_
    

def apply_arcsinh_to_bag_cello_df(
    cello_df: BagCelloDf,
    inplace: bool = False,
):
    if not inplace:
        new_cell_df = apply_arcsinh_to_df(df_=cello_df.cell_df, marker_col_names=cello_df.marker_col_names)
        return BagCelloDf(
            cell_df=new_cell_df,
            image_id_column=cello_df.image_id_column,
            distance_threshold=cello_df.distance_threshold,
            marker_col_names=cello_df.marker_col_names,
            pos_col_names=cello_df.pos_col_names,
            cellohood_neighborhood_cluster_colname=cello_df.cellohood_neighborhood_cluster_colname,
        )
    else:
        apply_arcsinh_to_df(df_=cello_df.cell_df, marker_col_names=cello_df.marker_col_names, inplace=True)
        return cello_df
