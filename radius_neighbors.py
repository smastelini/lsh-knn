from river import base

from lsh_buffer import LSHBuffer


class RadiusNeighborsRegressor(base.Regressor):
    """Radius Neighbors regressor.

    This non-parametric regression method keeps track of the last `window_size` training
    samples. Predictions are obtained by aggregating the values of the (closest) neighbors
    within the radius `r` from which query point. This implementation relies on an incrementally
    maintained Locality Sensitive Hashing Structure (LSH) to speed up nearest neighbor
    queries.

    Parameters
    ----------
    r
        The radius of the hypersphere for constructing. This parameter defines
        the r-neighborhood around each query point.
    max_size
        The maximum size of the window storing the last observed samples.
    p
        p-norm value for the Minkowski metric. When `p=1`, this corresponds to the
        Manhattan distance, while `p=2` corresponds to the Euclidean distance.
        Only `p=2` is currently supported.
    aggregation
        The method to aggregate the target values of neighbors.</br>
        - 'uniform': all points within distance `r` from the query point contribute equally
        to the responses.</br>
        - 'distance': the found points are weighted according to the inverse of their distances
        to the query point.
    k
        The number of random projections per hash table in the LSH scheme.
    delta
        The acceptable probability of failing to find a neighbor within distance `r`
        of the query point.
    w
        The quantization radius of the LSH scheme.

    Notes
    -----
    This estimator is not optimal for a mixture of categorical and numerical
    features. This implementation treats all features from a given stream as
    numerical.

    Only euclidean distances are currently supported, due to the underlying Locality Sensitive
    Hashing (LSH) structure used to query the nearest points.

    Examples
    --------


    """

    _UNIFORM = 'uniform'
    _DISTANCE = 'distance'

    def __init__(self, r: float = 1.0, max_size: int = 1000, p: float = 2,
                 aggregation: str = 'uniform', k: int = 3, delta: float = 0.1, w: float = 4,
                 seed: int = None):
        super().__init__()
        self.r = r
        self.max_size = max_size
        self.p = 2
        self.k = k
        self.delta = delta
        self.w = w
        self.seed = seed

        if aggregation not in [self._UNIFORM, self._DISTANCE]:
            raise ValueError(f'Invalid "aggregation" value: {aggregation}.\n'
                             f'Valid options are: {[self._UNIFORM, self._DISTANCE]}.')
        self.aggregation = aggregation

        self._buffer = LSHBuffer(R=self.r, max_size=self.max_size, k=self.k, delta=self.delta,
                                 w=self.w, seed=self.seed)

    def learn_one(self, x, y):  # noqa
        self._buffer.append((x, y))

        return self

    def predict_one(self, x):
        if self._buffer.size == 0:
            # Not enough information available, return default prediction
            return 0.

        dists, neighbors = self._buffer.query(x)
        if len(neighbors) == 0:
            # No near neighbors were found
            return 0.

        # If the closest neighbor has a distance of 0, then return its output
        if dists[0] == 0:
            return neighbors[0][1]

        if self.aggregation == self._UNIFORM:
            return sum(xy[1] for xy in neighbors) / len(neighbors)
        else:  # weighted mean
            return (
                sum(xy[1] / d for xy, d in zip(neighbors, dists)) /
                sum(1 / d for d in dists)
            )