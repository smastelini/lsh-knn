import collections
import functools
import math
import random

from river.utils import VectorDict


class LSHBuffer:
    def __init__(self, max_size: int = 1000, k: int = 3, L: int = 10, r: float = 4, seed=None):
        self.max_size = max_size
        self.k = k
        self.L = L
        self.r = r
        self.seed = seed

        self._size: int = 0
        self._next: int = 0  # Next position to add in the buffer
        self._buffer: list = [None for _ in range(self.max_size)]
        self._lsh = [collections.defaultdict(set) for _ in range(self.L)]

        self._projs = None
        self._rng = random.Random(self.seed)

    @property
    def size(self) -> int:
        return self._size

    def _init_projections(self, x):
        a_b = collections.namedtuple('Ab', ['a', 'b'])
        for h in range(self.L):
            self._projs[h] = {}
            for p in range(self.k):
                self._projs[h][p] = a_b(
                    a=VectorDict(data={f_id: self._rng.gauss(mu=0, sigma=1) for f_id in x}),
                    b=self._rng.uniform(0, self.r)
                )

    def _project(self, x):
        x_ = VectorDict(x)
        for h in range(self.L):
            projection = []
            for p in range(self.k):
                projection.append(
                    math.floor((self._projs[h][p].a @ x_ + self._projs[h][p].b) / self.r)
                )
            yield tuple(projection)

    def append(self, elem):
        x, y = elem

        if not self._projs:
            self._init_projections(x)

        if self._size == self.max_size:
            # TODO: element removal
            pass

        # Adds element to the buffer
        self._buffer[self._next] = elem
        for h, code in enumerate(self._project(x)):
            self._lsh[h][code].add(self._next)  # Add index to bucket

        # Update the circular buffer index
        self._next += 1 if self._next < self.max_size - 1 else 0

    def pop(self):
        pass

    def popleft(self):
        pass

    def query(self, x, n_neighbors=3, *, probes=1):
        pass

    def query_ball(self, x, eps=0.5, *, probes=1):
        pass
