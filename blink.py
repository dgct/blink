import numpy as np
import scipy.sparse as sp


BIOCHEM_SHIFTS = {
    "C": 12.0,
    "H": 1.00783,
    "H2": 2.01566,
    "O": 15.99491,
    "NH2-O": 0.02381,
    "PO3": 78.95851,
    "S": 31.97207,
}

ISOTOPIC_SHIFTS = {
    "13C-12C": 1.0034,
    "D-H": 1.0063,
    "15N-14N": 0.9970,
    "18O-16O": 2.0042,
}


class SparseData:
    def __init__(
        self,
        x,
        y=None,
        x_ends=None,
        x_kind="points",
        y_kind="values",
        value_power=0.5,
        tolerance=0.01,
        shifts=[],
        mirror_shifts=True,
        combination_steps=1,
        metadata=None,
    ):

        assert (y is None) or ([xi.shape for xi in x] == [yj.shape for yj in y])
        assert (x_ends is None) or (len(x) == len(x_ends))

        # flatten input data backfilling if necessary
        ids = np.concatenate([[i] * len(xi) for i, xi in enumerate(x)])
        x = np.concatenate(x)
        if y is None:
            y = np.ones_like(x)
        else:
            y = np.concatenate(y, axis=0)
        if x_ends is not None:
            x_ends = np.asarray(x_ends)

        # id data
        self._id = ids
        self.metadata = metadata

        # x data
        self._points = x
        self._x_ends = x_ends
        self._losses = None
        if x_kind == "points":
            self._x = self._points
        elif x_kind == "losses":
            self._x = self._losses_proxy

        # y data
        self._values = y**value_power
        self._value_power = value_power
        self._counts = None
        if y_kind == "values":
            self._y = self._values
        elif y_kind == "counts":
            self._y = self._counts_proxy

        # link data
        self._tolerance = tolerance
        self._shifts = np.asarray(shifts)
        self._mirror_shifts = mirror_shifts
        self._combination_steps = combination_steps
        self._kernel = None
        self._norm = None

        self._update()

    #############
    # Properties
    #############
    # X

    @property
    def _losses_proxy(self):
        if self._losses is None:
            self._losses = self._x_ends[self._id] - self._points
        return self._losses

    @property
    def x_kind(self):
        if self._x is self._points:
            return "points"
        elif self._x is self._losses:
            return "losses"

    @x_kind.setter
    def x_kind(self, new_x_kind):
        if (new_x_kind == "points") and (self._x is self._losses):
            self._x = self._points
            self._kernel = -(self._kernel - self._x_ends[self._id])
        elif (new_x_kind == "losses") and (self._x is self._points):
            self._x = self._losses_proxy
            self._kernel = self._x_ends[self._id] - self._kernel

    #############
    # Y

    @property
    def _counts_proxy(self):
        if self._counts is None:
            self._counts = self._values.astype(bool)
        return self._counts

    @property
    def y_kind(self):
        if self._y is self._values:
            return "values"
        elif self._y is self._counts:
            return "counts"

    @y_kind.setter
    def y_kind(self, new_y_kind):
        if (new_y_kind == "values") and (self._y is self._counts):
            self._y = self._values
            self._normalize()
        elif (new_y_kind == "counts") and (self._y is self._values):
            self._y = self._counts_proxy
            self._normalize()

    @property
    def value_power(self):
        return self._value_power

    @value_power.setter
    def value_power(self, new_value_power):
        if self._value_power != new_value_power:
            self._y = self._y ** (new_value_power / self._value_power)
            self._value_power = new_value_power
            self._normalize()

    #############
    # Link

    @property
    def tolerance(self):
        return self._tolerance

    @tolerance.setter
    def tolerance(self, new_tolerance):
        if self._tolerance != new_tolerance:
            self._tolerance = new_tolerance
            self._normalize()

    @property
    def shifts(self):
        return self._shifts

    @shifts.setter
    def shifts(self, new_shifts):
        if (self._shifts.shape != new_shifts.shape) or (
            self._shifts != new_shifts
        ).any():
            self._shifts = new_shifts
            self._update()

    @property
    def combination_steps(self):
        return self._combination_steps

    @combination_steps.setter
    def combination_steps(self, new_combination_steps):
        if self._combination_steps != new_combination_steps:
            self._combination_steps = new_combination_steps
            self._update()

    @property
    def mirror_shifts(self):
        return self._mirror_shifts

    @mirror_shifts.setter
    def mirror_shifts(self, mirror_shifts):
        if self._mirror_shifts != new_mirror_shifts:
            self._mirror_shifts = new_mirror_shifts
            self._update()

    #################
    # Helper Methods
    #################

    def _update(self):
        if self._mirror_shifts:
            shifts = np.multiply.outer(self._shifts, [1, -1]).flatten()
        else:
            shifts = self._shifts

        shifts = np.unique(np.append(0.0, shifts))

        # recursively combine shifts within a specified number of steps
        def combine(shifts, combination_steps):
            if combination_steps == 0:
                return np.array([0.0])
            if combination_steps == 1:
                return shifts
            else:
                return np.add.outer(shifts, combine(shifts, combination_steps - 1))

        shifts = np.unique(combine(shifts, self._combination_steps).flatten())

        self._kernel = np.add.outer(shifts, self._x)

        self._normalize()

    def _normalize(self):
        link = self._link(self, same_id=True)

        link = link[:, np.argsort(self._id[link[0]])]

        same_id = self._id[link[0]] == self._id[link[1]]

        id_mask = self._id[link[0]][1:] != self._id[link[0]][:-1]
        id_mask = np.append(True, id_mask)
        (id_edge,) = np.nonzero(id_mask)

        self._norm = (
            np.add.reduceat(
                same_id
                * self._parabolic_blur(self, link)
                * self._y[link[0]]
                * self._y[link[1]],
                id_edge,
            )
            ** -0.5
        )

        if self._y is self._counts:
            self._norm *= np.add.reduceat(self._counts[link[0]], id_edge) ** 0.5

    def _link(self, other, same_id=False):
        assert self._tolerance == other.tolerance

        self_x = self._x
        other_kernel = other._kernel
        tolerance = self._tolerance

        if same_id:
            assert self is other

            self_x = 1j * self_x + self._id
            other_kernel = 1j * other_kernel + self._id
            tolerance = 1j * tolerance

        sort_idx = np.argsort(self_x)

        overlap = np.array(
            [
                np.searchsorted(
                    self_x[sort_idx],
                    other_kernel.ravel() - tolerance,
                    "left",
                ),
                np.searchsorted(
                    self_x[sort_idx],
                    other_kernel.ravel() + tolerance,
                    "right",
                ),
            ]
        )

        link_idx = np.repeat(
            np.arange(overlap.shape[1]), overlap[1] - overlap[0], axis=0
        )

        link = np.array(
            [
                sort_idx[_multi_arange(overlap[:, overlap[0] != overlap[1]].T)],
                link_idx % other_kernel.shape[1],
                link_idx // other_kernel.shape[1],
            ]
        )

        return link

    def _parabolic_blur(self, other, link):
        x_diff = self._x[link[0]] - other._kernel[link[2], link[1]]

        return (3 / 4) * (1 - abs(x_diff / self._tolerance) ** 2)

    #################
    # Public Methods
    #################

    def score(self, other):
        assert self.x_kind == other.x_kind
        assert self.y_kind == other.y_kind

        assert self.tolerance == other.tolerance
        assert (self._shifts.shape == other._shifts.shape) and (
            self._shifts == other._shifts
        ).all()
        assert self.mirror_shifts == other.mirror_shifts
        assert self.combination_steps == other.combination_steps

        ordered = self._x.size <= other._x.size
        if not ordered:
            self, other = other, self

        link = self._link(other)
        x_bins = np.unique(link[0])

        S = sp.csr_matrix(
            (
                (self._norm[self._id] * self._y)[x_bins],
                (self._id[x_bins], x_bins),
            ),
        )

        O = sp.csr_matrix(
            (
                self._parabolic_blur(other, link)
                * (other._norm[other._id] * other._y)[link[1]],
                (link[0], other._id[link[1]]),
            ),
        )

        blink_score = S.dot(O)
        if not ordered:
            blink_score = blink_score.T

        return blink_score


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
