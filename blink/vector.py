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
        if np.size(x) == 0 and isinstance(y, list):
            x = np.concatenate([[i] * yy.shape[-1] for i, yy in enumerate(y)])
            y = np.concatenate(y, axis=-1)
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

        if data.ndim != 1 or data.dtype.kind not in "biuf":
            raise ValueError("data is mishapen, non-numeric, or complex")

        if not (len(x[0]) == len(y[0]) == len(data)):
            raise ValueError("x, y, and data array must all be same length")

        self.x_tolerance = x_tolerance
        self.y_tolerance = y_tolerance

        self._x = x
        self._y = y
        self._data = data

        self.clean()

        if shape is not None:
            self.shape = shape

    #################
    # Properties
    #################

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        value = np.array(value, copy=False, ndmin=2)

        if len(self.x[0]) != len(value[0]):
            raise ValueError("old x and new x must be same length")

        self._x = value
        self.clean()

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, value):
        value = np.array(value, copy=False, ndmin=2)

        if len(self.y[0]) != len(value[0]):
            raise ValueError("old y and new y must be same length")

        self._y = value
        self.clean()

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        if len(self.data) != len(value):
            raise ValueError("old data and new data must be same length")

        self._data = value

    #################
    # Blink Methods
    #################

    def _link(self, other):
        def _multi_arange(a):
            if a.shape[1] == 3:
                steps = a[:, 2]
            if a.size == 0:
                return np.array([], dtype=int)
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

    def _include(self, other, link, include):
        if isinstance(include, self.__class__):
            include = [include]

        out_coord = self.x[0, link[0]] + 1j * other.y[0, link[1]]
        include_coord = np.concatenate([i.x[0] + 1j * i.y[0] for i in include])

        new_link = link[:, np.isin(out_coord, include_coord)]

        return new_link

    def _exclude(self, other, link, exclude):
        if isinstance(exclude, self.__class__):
            exclude = [exclude]

        out_coord = self.x[0, link[0]] + 1j * other.y[0, link[1]]
        exclude_coord = np.concatenate([e.x[0] + 1j * e.y[0] for e in exclude])

        new_link = link[:, ~np.isin(out_coord, exclude_coord)]

        return new_link

    def _phase(self, link, axis):
        if axis == "y":
            mask0 = 0
            mask1 = link

        elif axis == "x":
            mask0 = link[2]
            mask1 = link[1]

        tolerance = self.__dict__[axis + "_tolerance"]
        axis = self.__dict__[axis]

        if tolerance == 0:
            phase = np.ones_like(axis[mask0, mask1], dtype=complex)
        else:
            phase = np.sin(0.5 * np.pi * axis[mask0, mask1] / tolerance, dtype=complex)
            phase += 1j * np.cos(0.5 * np.pi * axis[mask0, mask1] / tolerance)

        if not isinstance(mask0, int):
            phase = np.conj(phase)

        return phase

    def sum_duplicates(self):
        diff_x = self.x[0, 1:] != self.x[0, :-1]
        diff_y = self.y[0, 1:] != self.y[0, :-1]
        diff = diff_x | diff_y
        diff = np.append(True, diff)
        (diff_edge,) = np.nonzero(diff)

        self._x = self.x[:, diff]
        self._y = self.y[:, diff]
        self._data = np.add.reduceat(self.data, diff_edge, dtype=self.data.dtype)

    def eliminate_zeros(self):
        mask = ~np.isclose(self.data, 0)
        self._x = self.x[:, mask]
        self._y = self.y[:, mask]
        self._data = self.data[mask]

    def clean(self):
        sort_idx = np.argsort(self.y[0] + 1j * self.x[0])

        self._x = self.x[:, sort_idx]
        self._y = self.y[:, sort_idx]
        self._data = self.data[sort_idx]

        xmax, ymax = 0, 0
        if len(self.x[0]) > 0:
            xmax += self.x[0].max() + (self.x[0].dtype.kind in "iu")
        if len(self.y[0]) > 0:
            ymax += self.y[0].max() + (self.y[0].dtype.kind in "iu")
        self.shape = (xmax, ymax)

        if self.data.size > 0:
            self.sum_duplicates()
            self.eliminate_zeros()

    #################
    # Operators
    #################

    def _operate(self, other, func):
        if not isinstance(other, (bool, float, int)) and other.dtype.kind not in "biuf":
            raise ValueError("{} is not a scalar or array of shape data".format(other))
        self.data = func(self.data, other)

    def __eq__(self, other):
        result = np.isclose((self - other).data, 0).all()

        return result

    def __neq__(self, other):
        result = not (self == other)

        return result

    # def __lt__(self, other):
    #     result = self.copy()
    #     result._operate(other, lambda i, o: i < o)
    #     result.eliminate_zeros()
    #     return result

    # def __gt__(self, other):
    #     result = self.copy()
    #     result._operate(other, lambda i, o: i > o)
    #     result.eliminate_zeros()
    #     return result

    # def __le__(self, other):
    #     result = self.copy()
    #     result._operate(other, lambda i, o: i <= o)
    #     result.eliminate_zeros()
    #     return result

    # def __ge__(self, other):
    #     result = self.copy()
    #     result._operate(other, lambda i, o: i >= o)
    #     result.eliminate_zeros()
    #     return result

    def __add__(self, other):
        if isinstance(other, self.__class__):
            result = sum([self, other])

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

    def __matmul__(
        self,
        other,
        link=None,
        include=None,
        exclude=None,
    ):
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
        if include is not None:
            link = self._include(other, link, include)
        if exclude is not None:
            link = self._exclude(other, link, exclude)

        y_bins = np.unique(link[0])

        if len(link[0]) == 0:
            result = sp.coo_matrix((0, 0))
        else:
            left_phase = self._phase(y_bins, "y")
            left = sp.csr_matrix(
                (
                    left_phase * self.data[y_bins],
                    (y_bins, y_bins),
                ),
            )

            right_phase = other._phase(link, "x")
            right = sp.csr_matrix(
                (
                    right_phase * other.data[link[1]],
                    (link[0], link[1]),
                ),
            )

            result = left.dot(right).tocoo()

        result = self.__class__(
            self.x[:, result.row],
            other.y[:, result.col],
            result.data.real,
            self.x_tolerance,
            other.y_tolerance,
            (self.shape[0], other.shape[1]),
        )

        return result

    #################
    # Public Methods
    #################

    def copy(self, **kwargs):
        new_args = self.__dict__.copy()
        new_args.update(kwargs)

        result = self.__class__(**new_args)
        return result

    def xslice(self, *args):
        if len(args) == 1:
            mask = self.x[0] == args
        elif len(args) == 2:
            mask = np.ones_like(self.x[0], dtype=bool)
            if args[0] is not None:
                mask &= args[0] <= self.x[0]
            if args[1] is not None:
                mask &= self.x[0] < args[1]
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
                mask &= self.y[0] < args[1]
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

    def clip(self, *args):
        if len(args) == 1:
            mask = self.data >= args
        elif len(args) == 2:
            mask = np.ones_like(self.data, dtype=bool)
            if args[0] is not None:
                mask &= args[0] <= self.data
            if args[1] is not None:
                mask &= self.data <= args[1]
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

    def center(self):
        _, ind, inv, count = np.unique(
            self.x[0],
            return_index=True,
            return_inverse=True,
            return_counts=True,
        )

        means = sp.coo_matrix(
            (
                self.data / count[inv],
                (np.zeros_like(self.data, dtype=int), ind[inv]),
            )
        )
        means.sum_duplicates()
        means = means.data[inv]

        result = self.__class__(
            self.x,
            self.y,
            self.data - means,
            self.x_tolerance,
            self.y_tolerance,
            self.shape,
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

    def norm(self, kind="l2", chunk_size=100):
        def l2norm(self):
            link = self._link(self.T)
            same = self.x[0, link[0]] == self.T.y[0, link[1]]
            link = link[:, same]

            norm_ = self.__matmul__(self.T, link) ** -0.5

            # set vector norm to sqrt sum if vector is boolean
            # enabling vector multiplication to count "blurry" matches
            if self.data.dtype.kind == "b":
                same = self.y[0, link[0]] == self.T.x[0, link[1]]
                link = link[:, same]

                norm_ *= self.__matmul__(self.T, link).data ** 0.5

            norm_.y_tolerance = 0
            self.x_tolerance = 0

            return norm_ @ self

        def l1norm(self):
            _, ind, inv = np.unique(
                self.x[0],
                return_index=True,
                return_inverse=True,
            )

            sums = sp.coo_matrix(
                (
                    self.data,
                    (np.zeros_like(self.data, dtype=int), ind[inv]),
                )
            )
            sums.sum_duplicates()
            sums = sums.data[inv]

            result = self.__class__(
                self.x,
                self.y,
                self.data / sums,
                self.x_tolerance,
                self.y_tolerance,
                self.shape,
            )

            return result

        if kind == "l2":
            _norm = l2norm
        elif kind == "l1":
            _norm = l1norm
        elif kind == "l0":
            self = self.copy()
            self.data = self.data.astype(bool)
            _norm = l2norm
        else:
            raise NotImplementedError(
                "{}norm not implemented for blink vectors".format(kind)
            )

        result = sum(
            [
                _norm(self.xslice(i, i + chunk_size))
                for i in np.arange(0, self.shape[0], chunk_size)
            ]
        )

        return result

    def score(
        self,
        other,
        norm="l2",
        chunk_size=1000,
        include=None,
        exclude=None,
    ):
        if norm is not None:
            self = self.norm(norm)
            other = other.norm(norm)

        result = sum(
            [
                self.__matmul__(
                    other.xslice(i, i + chunk_size).T, include=include, exclude=exclude
                )
                for i in np.arange(0, other.shape[0], chunk_size)
            ]
        )

        return result

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
    if len(vectors) == 0:
        result = vector()
    else:
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
