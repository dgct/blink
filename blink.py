import numpy as np
import scipy.sparse as sp
from scipy.stats import rankdata
import copy


class vector:
    def __init__(
        self,
        x=np.array([]),
        y=np.array([]),
        data=None,
        x_tolerance=0.0,
        y_tolerance=0.0,
    ):

        x = np.array(x, copy=False, ndmin=2)
        y = np.array(y, copy=False, ndmin=2)
        if data is None:
            data = np.ones_like(x)
        data = np.asarray(data)

        for name, var in zip(["x", "y"], [x, y, data]):
            if var.ndim != 2 or var.dtype.kind not in "iuf":
                raise ValueError(
                    "{} is mishapen or not castable to 32 bit numeric".format(name)
                )

        if data.ndim != 1 or data.dtype.kind not in "biufc":
            raise ValueError("data is mishapen or non-numeric")

        if not (len(x[0]) == len(y[0]) == len(data)):
            raise ValueError("x, y, and data array must all be same length")

        self.x_tolerance = x_tolerance
        self.y_tolerance = y_tolerance

        sort_idx = np.argsort(y[0], kind="merge")
        self.x = x.astype(x.dtype.str[:2] + "4")[:, sort_idx]
        self.y = y.astype(y.dtype.str[:2] + "4")[:, sort_idx]
        self.data = data[sort_idx]

        self._squeeze()
        self._prune()

    #################
    # Blink Methods
    #################

    def _blur(self, other, link):
        diff = self.y[0, link[0]] - other.x[link[2], link[1]]

        if np.isclose(diff, 0).all():
            return 1

        return 1 - (diff / self.y_tolerance) ** 2

    def _link(self, other):
        overlap = np.array(
            [
                np.searchsorted(self.y[0], other.x.ravel() - self.y_tolerance, "left"),
                np.searchsorted(self.y[0], other.x.ravel() + self.y_tolerance, "right"),
            ]
        )

        link_idx = np.repeat(
            np.arange(overlap.shape[1]),
            overlap[1] - overlap[0],
            axis=0,
        )

        link = np.array(
            [
                _multi_arange(overlap[:, overlap[0] != overlap[1]].T),
                link_idx % other.x.shape[1],
                link_idx // other.x.shape[1],
            ]
        )

        return link

    def _squeeze(self):
        same_x = self.x[0, 1:] != self.x[0, :-1]
        same_y = self.y[0, 1:] != self.y[0, :-1]
        same = same_x | same_y
        same = np.append(True, same)
        (same_edge,) = np.nonzero(same)

        self.x = self.x[:, same]
        self.y = self.y[:, same]
        self.data = np.add.reduceat(
            self.data,
            same_edge,
        )

    def _prune(self):
        mask = self.data != 0
        self.x = self.x[:, mask]
        self.y = self.y[:, mask]
        self.data = self.data[mask]

    #################
    # Operators
    #################

    def _operate(self, other, func):
        if (
            not isinstance(other, (float, int, complex))
            and other.dtype.kind not in "iufc"
        ):
            raise ValueError("{} is not a scalar or array of shape data".format(other))
        self.data = func(self.data, other)

    def __eq__(self, other):
        result = True
        result &= (self.x == other.x).all()
        result &= (self.y == other.y).all()
        result &= (self.data == other.data).all()

        return result

    def __neq__(self, other):
        result = False
        result |= (self.x != other.x).any()
        result |= (self.y != other.y).any()
        result |= (self.data != other.data).any()

        return result

    def __add__(self, other):
        if isinstance(other, self.__class__):
            result = self.__class__(
                np.concatenate([self.x, other.x]),
                np.concatenate([self.y, other.y]),
                np.concatenate([self.data, other.data]),
                self.x_tolerance,
                self.y_tolerance,
            )

            return result

        result = self.copy()
        result._operate(other, lambda i, o: i + o)
        return result

    def __iadd__(self, other):
        self._operate(other, lambda i, o: i + o)
        return self

    def __mul__(self, other):
        if isinstance(other, self.__class__):
            return self @ other

        result = self.copy()
        result._operate(other, lambda i, o: i * o)
        return result

    def __rmul__(self, other):
        return self * other

    def __imul__(self, other):
        self._operate(other, lambda i, o: i * o)
        return self

    def __truediv__(self, other):
        result = self.copy()
        result._operate(other, lambda i, o: i / o)
        return result

    def __rtruediv__(self, other):
        return (self**-1) * other

    def __itruediv__(self, other):
        self._operate(other, lambda i, o: i / o)
        return self

    def __floordiv__(self, other):
        result = self.copy()
        result._operate(other, lambda i, o: i // o)
        return result

    def __ifloordiv__(self, other):
        self._operate(other, lambda i, o: i // o)
        return self

    def __pow__(self, other):
        result = self.copy()
        result._operate(other, lambda i, o: i**o)
        return result

    def __ipow__(self, other):
        self._operate(other, lambda i, o: i**o)
        return self

    def __matmul__(self, other, link=None):
        if not isinstance(other, self.__class__):
            raise NotImplementedError(
                "vector multiplication only defined between blink vectors"
            )

        if link is None:
            link = self._link(other)
        y_bins = np.unique(link[0])

        left = sp.csr_matrix(
            (
                self.data[y_bins],
                (self.x[0, y_bins].view(dtype=np.uint32), y_bins),
            ),
        )

        right = sp.csr_matrix(
            (
                self._blur(other, link) * other.data[link[1]],
                (link[0], other.y[0, link[1]].view(dtype=np.uint32)),
            ),
        )

        result = left.dot(right).tocoo()

        result = self.__class__(
            result.row.view(dtype=self.x.dtype),
            result.col.view(dtype=other.y.dtype),
            result.data,
            self.x_tolerance,
            other.y_tolerance,
        )

        return result

    #################
    # Public Methods
    #################

    def copy(self):
        return copy.copy(self)

    def transpose(self):
        result = self.__class__(
            self.y,
            self.x,
            self.data,
            self.y_tolerance,
            self.x_tolerance,
        )

        return result

    @property
    def T(self):
        return self.transpose()

    def diag(self):
        same = self.x[0] == self.y[0]

        result = self.__class__(
            self.x[:, same],
            self.y[:, same],
            self.data[same],
            self.x_tolerance,
            self.y_tolerance,
        )

        return result

    def norm(self):
        link = self._link(self.T)
        same = self.x[0, link[0]] == self.T.y[0, link[1]]
        link = link[:, same]

        norm_ = self.__matmul__(self.T, link) ** -0.5
        norm_.y_tolerance = 0

        return norm_ @ self

    def score(self, other, norm=False):
        if norm:
            self = self.norm()
            other = other.norm()

        return self @ other.T

    def tocoo(self):
        return sp.coo_matrix(
            (
                self.data,
                (self.x[0].view(np.uint32), self.y[0].view(np.uint32)),
            )
        )

    def toarray(self):
        result = self.copy()
        result.x = rankdata(result.x[0], "dense").astype(np.uint32)[None, :] - 1
        result.y = rankdata(result.y[0], "dense").astype(np.uint32)[None, :] - 1

        return result.tocoo().toarray()

    def save(self, file):
        np.savez(file, **self.__dict__)


def load(file):
    result = vector(**np.load(file))
    return result


def _multi_arange(a):
    if a.shape[1] == 3:
        steps = a[:, 2]
    else:
        steps = np.ones(a.shape[0], dtype=int)

    lens = ((a[:, 1] - a[:, 0]) + steps - np.sign(steps)) // steps
    b = np.repeat(steps, lens)
    ends = (lens - 1) * steps + a[:, 0]
    b[0] = a[0, 0]
    b[lens[:-1].cumsum()] = a[1:, 0] - ends[:-1]
    return b.cumsum()
