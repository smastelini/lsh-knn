cimport cython

import bisect
import collections
import math
import random

from scipy.stats import norm
import numpy as np

from river.utils import VectorDict
from river.utils.math import minkowski_distance


cdef class LSHBuffer:
    """

    Parameters
    ----------
    max_size
        The size of the buffer to store samples.
    R
        The radius of the hypersphere that defines the R-Nearest Neighbours around a query point.
    delta
        Acceptable probability of failing to return a "R"-neighbour for a query point. Hence,
        the probability of success is $1 - \\delta$.
    k
        The number of hash functions per table, i.e., the dimension of the projections.
    w
        The quantization radius.
    seed
        Random number generator seed for reproducibility.
    """

    cdef:
        readonly long max_size
        readonly long size
        readonly double R
        readonly double delta
        readonly long k
        readonly long L
        readonly double w
        readonly long p
        readonly long seed

        long _next
        long _oldest
        double _pr_col
        list _buffer
        list _lsh
        list _rprojs
        object _rng


    def __init__(self, max_size: int = 1000, R: float = 1.0, delta: float = 0.1, k: int = 3,
                 w: float = 4, p: int = 2, seed: int = None):
        self.max_size = max_size
        if not R > 0:
            raise ValueError(f'"R" must be greater than zero.')
        self.R = R

        self.delta = delta
        self.k = k
        self.w = w

        if not 1 <= p <= 2:
            raise ValueError(f'Invalid value of "p". It must be either "1" or "2".')
        self.p = p

        self._pr_col = 1. - 2. * norm.cdf(-self.w) - (
            (2. / (math.sqrt(2. * math.pi) * self.w))
            * (1. - math.exp(-(math.pow(self.w, 2)) / 2.))
        )
        self.L = <long> math.ceil(
            math.log(1. / self.delta) / (- math.log(1. - math.pow(self._pr_col, self.k)))
        )

        # Random number generators
        self.seed = seed
        self._rng = random.Random(self.seed)
        np.random.seed(self.seed)

        # Inner properties
        self.size = 0
        self._next = 0  # Next position to add in the buffer
        self._oldest = 0  # The oldest position in the buffer
        self._buffer = [None] * self.max_size  # The actual buffer
        self._lsh = [collections.defaultdict(set) for _ in range(self.L)]
        self._rprojs = None  # The random projections

    @property
    def success_probability(self) -> float:
        return 1 - (1 - math.pow(self._pr_col, self.k)) ** self.L

    cdef void _init_projections(self, dict x):
        """Initialize the random projections.

        Parameters
        ----------
        x
            Observation from which infer the dimension of the projections.
        """
        cdef Axb = collections.namedtuple('Ab', ['a', 'b'])
        self._rprojs = [None] * self.L
        if self.p == 1:
            for h in range(self.L):
                self._rprojs[h] = [None] * self.k
                for p in range(self.k):
                    # Initalize random projections by sampling from a standard Cauchy dist
                    self._rprojs[h][p] = Axb(
                        a=VectorDict(data={fid: np.random.standard_cauchy() for fid in x}),
                        b=self._rng.uniform(0, self.w)
                    )
        else:
            for h in range(self.L):
                self._rprojs[h] = [None] * self.k
                for p in range(self.k):
                    # Initalize random projections by sampling from a standard Gaussian dist
                    self._rprojs[h][p] = Axb(
                        a=VectorDict(data={fid: self._rng.gauss(mu=0, sigma=1) for fid in x}),
                        b=self._rng.uniform(0, self.w)
                    )

    cdef list _hash(self, dict x):
        """Generate the codes of a given observation for each of the L hash tables.

        Parameters
        ----------
        x
            Sample for which we want to calculate the hash code.
        """
        # Scale down x by factor R
        cdef x_ = VectorDict({i: x[i] / self.R for i in x})

        codes = [None] * self.L
        for h in range(self.L):
            codes[h] = tuple(
                <long> math.floor((self._rprojs[h][p].a @ x_ + self._rprojs[h][p].b) / self.w)
                for p in range(self.k)
            )

        return codes

    cdef void _add_to_hash(self, dict x, long index):
        # Save buffer index in the correct hash positions
        for h, code in enumerate(self._hash(x)):
            self._lsh[h][code].add(index)  # Add index to bucket

    cdef void _rem_from_hash(self, dict x, long index):
        for h, code in enumerate(self._hash(x)):
            self._lsh[h][code].discard(index)

            if len(self._lsh[h][code]) == 0:
                del self._lsh[h][code]

    cpdef void append(self, tuple elem):
        x, y = elem

        if not self._rprojs:
            self._init_projections(x)

        cdef bint slot_replaced = self._buffer[self._next] is not None

        # Remove previously stored element from the hash tables
        if slot_replaced:
            x_ = self._buffer[self._next][0]
            self._rem_from_hash(x_, self._next)

        # Adds element to the buffer
        self._buffer[self._next] = elem
        self._add_to_hash(x, self._next)

        # Update the circular buffer index
        self._next += 1 if self._next < self.max_size - 1 else 0

        if slot_replaced:
            self._oldest = self._next
        else:  # Actual buffer increased
            self.size += 1

    cpdef tuple pop(self):
        """Remove and return the most recent element added to the buffer."""
        if self.size > 0:
            self._next = self._next - 1 if self._next > 0 else self.max_size - 1
            x, y = self._buffer[self._next]

            # Update hash structure to remove the extracted observation
            self._rem_from_hash(x, self._next)

            # Update buffer size
            self.size -= 1

            return x, y

    cpdef tuple popleft(self):
        """Remove and return the oldest element in the buffer."""
        if self.size > 0:
            x, y = self._buffer[self._oldest]

            # Update hash structure to remove the extracted observation
            self._rem_from_hash(x, self._oldest)
            self._oldest = self._oldest + 1 if self._oldest < self.max_size - 1 else 0

            if self._oldest == self._next:
                # Shift circular buffer and make its starting point be the index 0
                self._oldest = self._next = 0

            # Update buffer size
            self.size -= 1

            return x, y

    cpdef tuple query(self, dict x, double max_points=-1, double p=2.):
        if max_points < 0:
            max_points = math.inf

        cdef:
            long count = 0
            set point_set = set()

        # Retrieve points
        for h, code in enumerate(self._hash(x)):
            point_set |= self._lsh[h][code]

            count += len(self._lsh[h][code])
            # Approximated search: stop once max_points are explored
            if count >= max_points:
                break

        cdef:
            list points = list()
            list distances = list()
            double dist
            long pos

        for q in point_set:
            dist = minkowski_distance(x, self._buffer[q][0], p=2)

            # Retrieve points ordered by their distance to the query point
            pos = bisect.bisect(distances, dist)
            distances.insert(pos, dist)
            points.insert(pos, self._buffer[q])

        return distances, points

    cpdef LSHBuffer clear(self):
        """Clear all stored elements."""
        self._next = 0
        self._oldest = 0
        self.size = 0
        self._buffer = [None] * self.max_size
        self._lsh = [collections.defaultdict(set) for _ in range(self.L)]

        return self
