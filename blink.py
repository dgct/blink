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
        x_transform=lambda x: np.array([x]),
        y_transform=lambda y: np.array([y]),
    ):

        x = np.asarray(x)
        y = np.asarray(y)
        if data is None:
            data = np.ones_like(x)
        data = np.asarray(data)

        for name, var in zip(["x", "y", "data"], [x, y, data]):
            if var.ndim != 1 or var.dtype.kind not in "biufcmM":
                raise ValueError("{} is mishapen or non-numeric.".format(name))

        if not (len(x) == len(y) == len(data)):
            raise ValueError("x, y, and data not equal length.")

        self.x_tolerance = x_tolerance
        self.y_tolerance = y_tolerance
        self.x_transform = x_transform
        self.y_transform = y_transform

        idx_sort = np.argsort(y + 1j * x)

        self.x = x_transform(x[idx_sort])
        self.y = y_transform(y[idx_sort])
        self.data = data[idx_sort]

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
                np.searchsorted(
                    self.y[0],
                    other.x.ravel() - self.y_tolerance,
                    "left",
                ),
                np.searchsorted(
                    self.y[0],
                    other.x.ravel() + self.y_tolerance,
                    "right",
                ),
            ]
        )

        link_idx = np.repeat(
            np.arange(overlap.shape[1]), overlap[1] - overlap[0], axis=0
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
        if not isinstance(other, (float, int, complex)):
            raise ValueError("{} is not a scalar.".format(other))
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
            if not (
                (self.x_tolerance == other.x_tolerance)
                and (self.y_tolerance == other.y_tolerance)
                and (self.x_transform == other.x_transform)
                and (self.y_transform == other.y_transform)
            ):
                raise ValueError("tolerances and transforms not equal between vectors.")

            result = self.__class__(
                np.concatenate([self.x[0], other.x[0]]),
                np.concatenate([self.y[0], other.y[0]]),
                np.concatenate([self.data, other.data]),
                self.x_tolerance,
                self.y_tolerance,
                self.x_transform,
                self.y_transform,
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

    def __matmul__(self, other):
        if not isinstance(other, self.__class__):
            raise NotImplementedError(
                "vector multiplication only defined between blink vectors"
            )
        if self.y_tolerance != other.x_tolerance:
            raise ValueError(
                "x tolerance of right vector must equal y tolerance of left vector"
            )

        link = self._link(other)

        _, linked = np.unique(link, axis=1, return_inverse=True)
        linked = linked[1:] != linked[:-1]
        linked = np.append(True, linked)
        (linked_edge,) = np.nonzero(linked)

        result = self.__class__(
            self.x[0, link[0]],
            other.y[0, link[1]],
            np.add.reduceat(
                self._blur(other, link) * self.data[link[0]] * other.data[link[1]],
                linked_edge,
            ),
            self.x_tolerance,
            other.y_tolerance,
            self.x_transform,
            other.y_transform,
        )

        return result

    #################
    # Public Methods
    #################

    def copy(self):
        return copy.copy(self)

    def transpose(self):
        result = self.__class__(
            self.y[0],
            self.x[0],
            self.data,
            self.y_tolerance,
            self.x_tolerance,
            self.y_transform,
            self.x_transform,
        )

        return result

    @property
    def T(self):
        return self.transpose()

    def diag(self):
        same = self.x[0] == self.y[0]

        result = self.__class__(
            self.x[0, same],
            self.y[0, same],
            self.data[same],
            self.x_tolerance,
            self.y_tolerance,
            self.x_transform,
            self.y_transform,
        )

        return result

    def norm(self):
        result = (self @ self.T).diag()

        result **= -0.5
        result.y_tolerance = 0
        self.x_tolerance = 0

        result = result @ self

        return result

    def score(self, other, norm=False):
        if norm:
            self = self.norm()
            other = other.norm()

        return self @ other.T

    def tocoo(self):
        result = sp.coo_matrix(
            (
                self.data,
                (rankdata(self.x[0], "dense") - 1, rankdata(self.y[0], "dense") - 1),
            )
        )
        return result

    def toarray(self):
        return self.tocoo().toarray()


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
