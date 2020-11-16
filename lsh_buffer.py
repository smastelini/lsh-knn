import collections
import functools
import math
import random
import typing
from scipy.stats import norm

from river.utils import VectorDict


class LSHBuffer:
    """

    Parameters
    ----------
    max_size
    k
        The number of hash functions per table.
    r
        The quantization radius.
    c
        The approximation factor.
    delta
        Probability of not returning a "r"-neighbour. Hence, the probability of success is
        $1 - \\delta$.
    mode
        exact, approximate
    seed
    """

    def __init__(self, max_size: int = 1000, k: int = 10, r: float = 4, c: float = 1.5,
                 delta: float = 0.1, seed=None):
        self.max_size = max_size
        self.k = k
        self.r = r
        self.c = c  # TODO verify later
        self.p1 = 1 - 2 * norm.cdf(-self.r) - ((2 / (math.sqrt(2 * math.pi) * self.r))
                           * (1 - math.exp(-(self.r ** 2) / 2)))
        # self.p2 = ...
        self.L = math.ceil(math.log(1 / delta) / (- math.log(1 - self.p1 ** self.k)))
        self.seed = seed

        self._size: int = 0
        self._next: int = 0  # Next position to add in the buffer
        self._oldest: int = 0  # The oldest position in the buffer
        self._buffer: list = [None for _ in range(self.max_size)]
        self._lsh = [collections.defaultdict(set) for _ in range(self.L)]

        self._rprojs = None
        self._rng = random.Random(self.seed)

    @property
    def size(self) -> int:
        return self._size

    def _init_projections(self, x):
        """Initialize the random projections.

        Parameters
        ----------
        x
            Observation from which infer the dimension of the projections.
        """
        Axb = collections.namedtuple('Ab', ['a', 'b'])
        for h in range(self.L):
            self._rprojs[h] = {}
            for p in range(self.k):
                self._rprojs[h][p] = Axb(
                    a=VectorDict(data={fid: self._rng.gauss(mu=0, sigma=1) for fid in x}),
                    b=self._rng.uniform(0, self.r)
                )

    def _hash(self, x):
        """Generate the codes of a given observation for each of the L hash tables.

        Parameters
        ----------
        x
            Sample for which we want to calculate the hash code.
        """
        x_ = VectorDict(x)
        for h in range(self.L):
            projection = []
            for p in range(self.k):  # TODO use map here?
                projection.append(
                    math.floor((self._rprojs[h][p].a @ x_ + self._rprojs[h][p].b) / self.r)
                )
            yield tuple(projection)

    def _rem_from_hash(self, x, index):
        for h, code in enumerate(self._hash(x)):
            self._lsh[h][code].discard(index)

    def append(self, elem):
        x, y = elem

        if not self._rprojs:
            self._init_projections(x)

        slot_replaced = self._buffer[self._next] is not None

        # Remove previously stored element from the hash tables
        if slot_replaced:
            x, = self._buffer[self._next]
            self._rem_from_hash(x, self._next)

        # Adds element to the buffer
        self._buffer[self._next] = elem

        # Save buffer index in the correct hash positions
        for h, code in enumerate(self._hash(x)):
            self._lsh[h][code].add(self._next)  # Add index to bucket

        # Update the circular buffer index
        self._next += 1 if self._next < self.max_size - 1 else 0

        if slot_replaced:
            self._oldest = self._next
        else:  # Actual buffer increased
            self._size += 1

    def pop(self):
        """Remove and return the most recent element added to the buffer."""
        if self.size > 0:
            self._next = self._next - 1 if self._next > 0 else self.max_size - 1
            x, y = self._buffer[self._next]

            # Update hash structure to remove the extracted observation
            self._rem_from_hash(x, self._next)

            # Update buffer size
            self._size -= 1

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
            self._size -= 1

            return x, y

    def query(self, x, n_neighbors=3, *, probes=1):
        pass

    def query_ball(self, x, eps=0.5, *, probes=1):
        pass
