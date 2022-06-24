import numpy as np
import scipy.sparse as sp


class vector:
    def __init__(
        self,
        x=np.array([]),
        y=np.array([]),
        data=None,
        x_tolerance=0.0,
        y_tolerance=0.0,
        shape=None,
    ):

        # default to rows if x is empty and y is list of numpy arrays
        if (x.size == 0) and isinstance(y, list):
            x = np.concatenate([[i] * len(yy) for i, yy in enumerate(y)])
            y = np.concatenate(y)
            if isinstance(data, list):
                data = np.concatenate(data)

        x = np.array(x, copy=False, ndmin=2)
        y = np.array(y, copy=False, ndmin=2)
        if data is None:
            data = np.ones_like(x[0], dtype=bool)
        data = np.asarray(data)

        for name, var in zip(["x", "y"], [x, y]):
            if var.ndim != 2:
                raise ValueError("{} is mishapen".format(name))

        if data.ndim != 1 or data.dtype.kind not in "biufc":
            raise ValueError("data is mishapen or non-numeric")

        if not (len(x[0]) == len(y[0]) == len(data)):
            raise ValueError("x, y, and data array must all be same length")

        self.x_tolerance = x_tolerance
        self.y_tolerance = y_tolerance

        sort_idx = np.argsort(y[0])
        self.x = x[:, sort_idx]
        self.y = y[:, sort_idx]
        self.data = data[sort_idx]

        if shape is None:
            xmax = self.x[0].max() + (self.x[0].dtype.kind in "iu")
            ymax = self.y[0].max() + (self.y[0].dtype.kind in "iu")
            shape = (xmax, ymax)
        self.shape = shape

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
        self.data = np.add.reduceat(self.data, same_edge)

    def _prune(self):
        mask = ~np.isclose(self.data, 0)
        self.x = self.x[:, mask]
        self.y = self.y[:, mask]
        self.data = self.data[mask]

    #################
    # Operators
    #################

    def _operate(self, other, func):
        if (
            not isinstance(other, (bool, float, int, complex))
            and other.dtype.kind not in "biufc"
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
                np.concatenate([self.x, other.x], axis=1),
                np.concatenate([self.y, other.y], axis=1),
                np.concatenate([self.data, other.data]),
                self.x_tolerance,
                self.y_tolerance,
                (
                    max(self.shape[0], other.shape[0]),
                    max(self.shape[1], other.shape[1]),
                ),
            )

            return result

        result = self.copy()
        result._operate(other, lambda i, o: i + o)
        return result

    def __sub__(self, other):
        return self + (-1 * other)

    def __iadd__(self, other):
        self._operate(other, lambda i, o: i + o)
        return self

    def __isub__(self, other):
        self._operate(other, lambda i, o: i - o)
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

        if self.y_tolerance != other.x_tolerance:
            raise ValueError(
                "y tolerance of left vector must equal x tolerance of right vector"
            )

        if link is None:
            link = self._link(other)
        y_bins = np.unique(link[0])

        left = sp.csr_matrix(
            (
                self.data[y_bins],
                (y_bins, y_bins),
            ),
        )

        right = sp.csr_matrix(
            (
                self._blur(other, link) * other.data[link[1]],
                (link[0], link[1]),
            ),
        )

        result = left.dot(right).tocoo()

        result = self.__class__(
            self.x[:, result.row],
            other.y[:, result.col],
            result.data,
            self.x_tolerance,
            other.y_tolerance,
            (self.shape[0], other.shape[1]),
        )

        return result

    #################
    # Public Methods
    #################

    def copy(self):
        result = self.__class__(**self.__dict__.copy())
        return result

    def xslice(self, *args):
        if len(args) == 1:
            mask = self.x[0] == args
        elif len(args) == 2:
            mask = np.ones_like(self.x[0], dtype=bool)
            if args[0] is not None:
                mask &= args[0] <= self.x[0]
            if args[1] is not None:
                mask &= self.x[0] <= args[1]
        else:
            raise ValueError("args must be on length 1 or 2")

        result = self.__class__(
            self.x[:, mask],
            self.y[:, mask],
            self.data[mask],
            self.x_tolerance,
            self.y_tolerance,
        )
        return result

    def yslice(self, *args):
        if len(args) == 1:
            mask = self.y[0] == args
        elif len(args) == 2:
            mask = np.ones_like(self.y[0], dtype=bool)
            if args[0] is not None:
                mask &= args[0] <= self.y[0]
            if args[1] is not None:
                mask &= self.y[0] <= args[1]
        else:
            raise ValueError("args must be on length 1 or 2")

        result = self.__class__(
            self.x[:, mask],
            self.y[:, mask],
            self.data[mask],
            self.x_tolerance,
            self.y_tolerance,
        )
        return result

    def transpose(self):
        result = self.__class__(
            self.y,
            self.x,
            self.data,
            self.y_tolerance,
            self.x_tolerance,
            (self.shape[1], self.shape[0]),
        )

        return result

    def conj(self):
        result = self.__class__(
            self.x,
            self.y,
            self.data.conj(),
            self.x_tolerance,
            self.y_tolerance,
            self.shape,
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
            self.shape,
        )

        return result

    def norm(self):
        link = self._link(self.conj().T)
        same = self.x[0, link[0]] == self.T.y[0, link[1]]
        link = link[:, same]

        norm_ = self.__matmul__(self.conj().T, link) ** -0.5

        # set vector norm to sqrt sum if vector is boolean
        # enabling vector multiplication to count "blurry" matches
        if self.data.dtype.kind == "b":
            same = self.y[0, link[0]] == self.T.x[0, link[1]]
            link = link[:, same]

            norm_ *= self.__matmul__(self.conj().T, link).data ** 0.5

        norm_.y_tolerance = 0
        self.x_tolerance = 0

        return norm_ @ self

    def score(self, other):
        self = self.norm()
        other = other.norm()

        return self @ other.conj().T

    def tocoo(self):
        return sp.coo_matrix(
            (
                self.data,
                (self.x[0], self.y[0]),
            ),
            shape=self.shape,
        )

    def toarray(self):
        return self.tocoo().toarray()

    def save(self, file):
        np.savez(file, **self.__dict__)


def load(file):
    result = vector(**np.load(file))
    return result


def sum(vectors):
    result = vector(
        np.concatenate([v.x for v in vectors], axis=1),
        np.concatenate([v.y for v in vectors], axis=1),
        np.concatenate([v.data for v in vectors]),
        np.max([v.x_tolerance for v in vectors]),
        np.max([v.y_tolerance for v in vectors]),
        (
            np.max([v.shape[0] for v in vectors]),
            np.max([v.shape[1] for v in vectors]),
        ),
    )

    return result
