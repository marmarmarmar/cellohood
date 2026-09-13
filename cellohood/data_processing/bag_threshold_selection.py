from collections import namedtuple

import numpy

from cellohood.data_processing.data_transforms import transform_df_to_bag_cello_df


BagThresholdSelectionResult = namedtuple(
    'BagThresholdSelectionResult',
    [
        'selected_threshold',
        'threshold_to_nb_bags',
        'threshold_to_sizes',
    ]
)


def _kneedle_knee(x, y):
    """
    Dependency-free 'Kneedle' knee detector (Satopaa et al., 2011) for a
    convex, decreasing curve: normalize both axes to [0, 1], then find the
    point of maximum vertical distance from the straight line connecting the
    curve's endpoints. That point is the 'knee' -- the spot where further
    movement along x stops paying for itself in y.
    """
    x = numpy.asarray(x, dtype=float)
    y = numpy.asarray(y, dtype=float)
    x_norm = (x - x.min()) / (x.max() - x.min())
    y_norm = (y - y.min()) / (y.max() - y.min())
    line_y = y_norm[0] + (y_norm[-1] - y_norm[0]) * (x_norm - x_norm[0]) / (x_norm[-1] - x_norm[0])
    distances = line_y - y_norm
    return int(numpy.argmax(distances))


def select_bag_distance_threshold(
    df_,
    image_id_column,
    pos_col_names,
    thresholds,
    marker_col_names=(),
    cellohood_neighborhood_cluster_colname='cellohood_neighborhood_cluster',
):
    """
    Data-driven heuristic for choosing the cell-bag distance threshold.

    Sweeps `thresholds` and, for each one, constructs cell bags via the same
    hierarchical-clustering procedure used elsewhere in Cellohood, tracking
    the number of resulting bags (cellular neighborhoods). This curve is a
    consistent, interpretable diminishing-returns shape: a steep decline at
    small thresholds, where growing the radius rapidly merges many trivially
    small bags, followed by a long, shallow tail, where further growth only
    slowly reduces the neighborhood count by merging already-substantial
    neighborhoods together.

    Growing the threshold trades two costs against each other:
    - Too small: bags carry almost no local compositional information (in
      the extreme, singleton bags carry none at all) -- an uninformative
      unit for the permutation-invariant encoder to learn from.
    - Too large: fewer independent bags remain to train/evaluate on
      (reduced statistical power), self-attention cost grows quadratically
      in bag size, and spatially distinct niches risk being merged together
      (over-smoothing).

    We locate the knee of the number-of-neighborhoods-vs-threshold curve
    with the Kneedle algorithm: the threshold past which further increases
    stop paying for themselves in fewer/larger bags. This introduces no
    free parameters beyond the swept threshold range itself.

    Args:
        df_: the cell dataframe (same input `transform_df_to_bag_cello_df`
            expects).
        image_id_column: see `transform_df_to_bag_cello_df`.
        pos_col_names: see `transform_df_to_bag_cello_df`.
        thresholds: an iterable of candidate distance thresholds to sweep.
        marker_col_names: see `transform_df_to_bag_cello_df` (not needed for
            the selection itself, only forwarded for bag construction).
        cellohood_neighborhood_cluster_colname: see `transform_df_to_bag_cello_df`.

    Returns:
        A `BagThresholdSelectionResult` with the selected threshold and the
        full swept curves, for inspection/plotting.
    """
    threshold_to_sizes = {}
    for threshold in thresholds:
        bag_cello_df = transform_df_to_bag_cello_df(
            df_=df_,
            image_id_column=image_id_column,
            pos_col_names=pos_col_names,
            distance_threshold=threshold,
            marker_col_names=list(marker_col_names),
        )
        sizes = bag_cello_df.cell_df.groupby(cellohood_neighborhood_cluster_colname).size().values
        threshold_to_sizes[threshold] = sizes

    threshold_to_nb_bags = {t: len(sizes) for t, sizes in threshold_to_sizes.items()}
    thresholds_arr = numpy.array(list(threshold_to_nb_bags.keys()))
    nb_bags_arr = numpy.array(list(threshold_to_nb_bags.values()))
    knee_idx = _kneedle_knee(thresholds_arr, nb_bags_arr)
    selected_threshold = thresholds_arr[knee_idx]

    return BagThresholdSelectionResult(
        selected_threshold=selected_threshold,
        threshold_to_nb_bags=threshold_to_nb_bags,
        threshold_to_sizes=threshold_to_sizes,
    )
