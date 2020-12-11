import warnings

from river import base
from lsh_buffer import LSHBuffer


class KNNRegressor(base.Regressor):
    """k-Nearest Neighbors regressor.

    This non-parametric regression method keeps track of the last
    `window_size` training samples. Predictions are obtained by
    aggregating the values of the closest n_neighbors stored-samples with
    respect to a query sample.

    Parameters
    ----------
    n_neighbors
        The number of nearest neighbors to search for.
    window_size
        The maximum size of the window storing the last observed samples.
    p
        p-norm value for the Minkowski metric. When `p=1`, this corresponds to the
        Manhattan distance, while `p=2` corresponds to the Euclidean distance.
        Can be either `1` or `2`.
    aggregation_method
        The method to aggregate the target values of neighbors.</br>
        - 'mean'</br>
        - 'weighted_mean'
    r
        The radius of the hypersphere for constructing the LSH structure. This parameter defines
        the r-neighborhood around each query point.
    c
        The approximation factor. Points within the distance `c * r` of a query point are
        guaranteed to be found with probability `1 - delta`.
    k
        The number of random projections per hash table in the LSH scheme.
    delta
        The acceptable probability of failing to find a neighbor within distance `c * r`
        of the query point.
    w
        The quantization radius of the LSH scheme.
    seed
        Random number generator seed for reproducibility.

    Notes
    -----
    This estimator is not optimal for a mixture of categorical and numerical
    features. This implementation treats all features from a given stream as
    numerical.

    Examples
    --------
    >>> from river import datasets
    >>> from river import evaluate
    >>> from river import metrics
    >>> from river import preprocessing
    >>> from k_nearest_neighbors import KNNRegressor
    >>> dataset = datasets.TrumpApproval()
    >>> model = (
    ...     preprocessing.StandardScaler() |
    ...     KNNRegressor(window_size=100, r=3., seed=42)
    ... )
    >>> metric = metrics.MAE()
    >>> evaluate.progressive_val_score(dataset, model, metric)
    MAE: 1.324855

    """

    _MEAN = 'mean'
    _WEIGHTED_MEAN = 'weighted_mean'
    _VALID = [_MEAN, _WEIGHTED_MEAN]

    def __init__(self, n_neighbors: int = 5, window_size: int = 1000, p: int = 2,
                 aggregation_method: str = 'mean', r: float = 1, c: float = 2, k: int = 3,
                 delta: float = 0.1, w: float = 4, seed: int = None):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.window_size = window_size
        self.p = p

        if aggregation_method not in self._VALID:
            raise ValueError(f'Invalid aggregation_method: {aggregation_method}.\n'
                             f'Valid options are: {self._VALID}')
        self.aggregation_method = aggregation_method

        self.r = r
        self.c = c
        self.k = k
        self.delta = delta
        self.w = w
        self.seed = seed

        self._buffer = LSHBuffer(max_size=self.window_size, R=self.r, c=self.c, k=self.k,
                                 delta=self.delta, w=self.w, p=self.p, seed=self.seed)

    def learn_one(self, x, y):
        """Update the model with a set of features `x` and a real target value `y`.

        Parameters
        ----------
        x
            A dictionary of features.
        y
            A numeric target.

        Returns
        -------
            self

        Notes
        -----
        For the K-Nearest Neighbors regressor, fitting the model is the
        equivalent of inserting the newer samples in the observed window,
        and if the `window_size` is reached, removing older results.

        """

        self._buffer.append((x, y))

        return self

    def predict_one(self, x):
        """Predict the target value of a set of features `x`.

        Search the KDTree for the `n_neighbors` nearest neighbors.

        Parameters
        ----------
        x
            A dictionary of features.

        Returns
        -------
            The prediction.

        """

        if self._buffer.size == 0:
            # Not enough information available, return default prediction
            return 0.

        dists, neighbors = self._buffer.query(x)

        if len(neighbors) == 0:
            if self._buffer.size == self.window_size:
                warnings.warn(
                    'No nearest neighbors were found in the approximate search.\n'
                    'You might try to increase the value of "r" or "c".',
                    category=RuntimeWarning
                )
            return 0.

        # If the closest neighbor has a distance of 0, then return it's output
        if dists[0] == 0:
            return neighbors[0][1]

        if len(neighbors) < self.n_neighbors:
            if self._buffer.size == self.window_size:
                warnings.warn(
                    'The number of neighbors found was smaller than "n_neighbors".\n'
                    'You might want trying to increase the value of "r" or "c" to avoid that.',
                    category=RuntimeWarning
                )
        else:
            neighbors = neighbors[:self.n_neighbors]
            dists = dists[:self.n_neighbors]

        if self.aggregation_method == self._MEAN:
            return sum(neighbor[1] for neighbor in neighbors) / len(neighbors)
        else:  # weighted mean
            return (
                sum(xy[1] / d for xy, d in zip(neighbors, dists)) / sum(1 / d for d in dists)
            )


class KNNClassifier(base.Classifier):
    """k-Nearest Neighbors classifier.

    This non-parametric regression method keeps track of the last
    `window_size` training samples. Predictions are obtained by
    aggregating the values of the closest n_neighbors stored-samples with
    respect to a query sample.

    Parameters
    ----------
    n_neighbors
        The number of nearest neighbors to search for.
    window_size
        The maximum size of the window storing the last observed samples.
    p
        p-norm value for the Minkowski metric. When `p=1`, this corresponds to the
        Manhattan distance, while `p=2` corresponds to the Euclidean distance.
        Can be either `1` or `2`.
    r
        The radius of the hypersphere for constructing the LSH structure. This parameter defines
        the r-neighborhood around each query point.
    c
        The approximation factor. Points within the distance `c * r` of a query point are
        guaranteed to be found with probability `1 - delta`.
    k
        The number of random projections per hash table in the LSH scheme.
    delta
        The acceptable probability of failing to find a neighbor within distance `c * r`
        of the query point.
    w
        The quantization radius of the LSH scheme.
    weighted
        If True, weight the contribution of each neighbor by it's inverse
        distance.
    seed
        Random number generator seed for reproducibility.

    Notes
    -----
    This estimator is not optimal for a mixture of categorical and numerical
    features. This implementation treats all features from a given stream as
    numerical.

    Examples
    --------
    >>> from river import datasets
    >>> from river import evaluate
    >>> from river import metrics
    >>> from river import preprocessing
    >>> from k_nearest_neighbors import KNNClassifier

    >>> dataset = datasets.Phishing()
    >>> model = KNNClassifier(c=2.0, k=4, seed=42)
    >>> metric = metrics.Accuracy()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    Accuracy: 92.47%

    """

    def __init__(self, n_neighbors: int = 5, window_size: int = 1000, p: int = 2,
                 r: float = 1, c: float = 2, k: int = 3, delta: float = 0.1,
                 w: float = 4, weighted=True, seed: int = None):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.window_size = window_size
        self.p = p
        self.r = r
        self.c = c
        self.k = k
        self.delta = delta
        self.w = w
        self.weighted = weighted
        self.seed = seed

        self._classes = set()
        self._buffer = LSHBuffer(max_size=self.window_size, R=self.r, c=self.c, k=self.k,
                                 delta=self.delta, w=self.w, p=self.p, seed=self.seed)

    def learn_one(self, x: dict, y: base.typing.ClfTarget):
        """Update the model with a set of features `x` and a real target value `y`.

        Parameters
        ----------
        x
            A dictionary of features.
        y
            A numeric target.

        Returns
        -------
            self

        Notes
        -----
        For the K-Nearest Neighbors regressor, fitting the model is the
        equivalent of inserting the newer samples in the observed window,
        and if the `window_size` is reached, removing older results.

        """
        self._classes.add(y)
        self._buffer.append((x, y))

        return self

    def predict_proba_one(self, x: dict):
        """Predict the probability of each label for a dictionary of features `x`.

        Parameters
        ----------
        x
            A dictionary of features.

        Returns
        -------
        proba
            A dictionary which associates a probability which each label.

        """

        proba = {class_idx: 0. for class_idx in self._classes}
        if self._buffer.size == 0:
            # Not enough information available, return default prediction
            return proba

        dists, neighbors = self._buffer.query(x)

        if len(neighbors) == 0:
            if self._buffer.size == self.window_size:
                warnings.warn(
                    'No nearest neighbors were found in the approximate search.\n'
                    'You might try to increase the value of "r" or "c".',
                    category=RuntimeWarning
                )
            return proba

        # If the closest neighbor has a distance of 0, then return it's output
        if dists[0] == 0:
            proba[neighbors[0][1]] = 1.
            return proba

        if len(neighbors) < self.n_neighbors:
            if self._buffer.size == self.window_size:
                warnings.warn(
                    'The number of neighbors found was smaller than "n_neighbors".\n'
                    'You might want trying to increase the value of "r" or "c" to avoid that.',
                    category=RuntimeWarning
                )
        else:
            neighbors = neighbors[:self.n_neighbors]
            dists = dists[:self.n_neighbors]

        if not self.weighted:  # Uniform weights
            for _, y_nn in neighbors:
                proba[y_nn] += 1.
        else:  # Use the inverse of the distance to weight the votes
            for d, (_, y_nn) in zip(dists, neighbors):
                proba[y_nn] += 1. / d

    @property
    def _multiclass(self):
        return True
