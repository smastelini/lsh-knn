import bisect
import collections
import math
import random
from scipy.stats import norm

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
        The number of hash functions per table.
    w
        The quantization radius.
    seed
    """

    cdef readonly long max_size
    cdef readonly long size
    cdef readonly double R
    cdef readonly double delta
    cdef readonly long k
    cdef readonly long L
    cdef readonly double w
    cdef readonly long seed

    cdef long _next
    cdef long _oldest
    cdef double _pr_col
    cdef list _buffer
    cdef list _lsh
    cdef dict _rprojs
    cdef _rng


    def __init__(self, max_size: int = 1000, R: float = 1.0, delta: float = 0.1, k: int = 3,
                 w: float = 4, seed: int = None):
        self.max_size = max_size
        self.k = k
        self.w = w

        if not R > 0:
            raise ValueError(f'"R" must be greater than zero.')
        self.R = R

        self._pr_col = 1. - 2. * norm.cdf(-self.w) - (
            (2. / (math.sqrt(2. * math.pi) * self.w))
            * (1. - math.exp(-(math.pow(self.w, 2)) / 2.))
        )
        self.L = <long> math.ceil(
            math.log(1. / delta) / (- math.log(1. - math.pow(self._pr_col, self.k)))
        )
        self.seed = seed

        self.size = 0
        self._next = 0  # Next position to add in the buffer
        self._oldest = 0  # The oldest position in the buffer
        self._buffer = [None for _ in range(self.max_size)]
        self._lsh = [collections.defaultdict(set) for _ in range(self.L)]

        self._rprojs = None
        self._rng = random.Random(self.seed)

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
        Axb = collections.namedtuple('Ab', ['a', 'b'])
        self._rprojs = {}
        for h in range(self.L):
            self._rprojs[h] = {}
            for p in range(self.k):
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
        cdef list codes = []
        for h in range(self.L):
            codes.append(
                tuple(
                    math.floor((self._rprojs[h][p].a @ x_ + self._rprojs[h][p].b) / self.w)
                    for p in range(self.k)
                )
            )

        return codes

    cdef void _add2hash(self, dict x, long index):
        # Save buffer index in the correct hash positions
        for h, code in enumerate(self._hash(x)):
            self._lsh[h][code].add(index)  # Add index to bucket

    cdef void _rem_from_hash(self, x, index):
        for h, code in enumerate(self._hash(x)):
            self._lsh[h][code].discard(index)

            if len(self._lsh[h][code]) == 0:
                del self._lsh[h][code]

    def append(self, elem):
        x, y = elem

        if not self._rprojs:
            self._init_projections(x)

        slot_replaced = self._buffer[self._next] is not None

        # Remove previously stored element from the hash tables
        if slot_replaced:
            x, _ = self._buffer[self._next]
            self._rem_from_hash(x, self._next)

        # Adds element to the buffer
        self._buffer[self._next] = elem
        self._add2hash(x, self._next)

        # Update the circular buffer index
        self._next += 1 if self._next < self.max_size - 1 else 0

        if slot_replaced:
            self._oldest = self._next
        else:  # Actual buffer increased
            self.size += 1

    def pop(self):
        """Remove and return the most recent element added to the buffer."""
        if self.size > 0:
            self._next = self._next - 1 if self._next > 0 else self.max_size - 1
            x, y = self._buffer[self._next]

            # Update hash structure to remove the extracted observation
            self._rem_from_hash(x, self._next)

            # Update buffer size
            self.size -= 1

            return x, y

    def popleft(self):
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

    def query(self, x, eps=None, *, p=2):
        if eps is None:
            eps = math.inf

        # Retrieve points
        point_set = set()
        for h, code in enumerate(self._hash(x)):
            point_set |= self._lsh[h][code]

        points = []
        distances = []
        for q in point_set:
            x_q, _ = self._buffer[q]
            dist = minkowski_distance(x, x_q, p=2)

            # Skip points whose distance to x is greater than eps
            if dist > eps:
                continue

            # Retrieve points ordered by their distance to the query point
            pos = bisect.bisect(distances, dist)
            distances.insert(pos, dist)
            points.insert(pos, self._buffer[q])

        return distances, points

    def clear(self) -> 'LSHBuffer':
        """Clear all stored elements."""
        self._next = 0
        self._oldest = 0
        self.size = 0
        self._buffer = [None for _ in range(self.max_size)]
        self._lsh = [collections.defaultdict(set) for _ in range(self.L)]

        return self
