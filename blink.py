import numpy as np
import scipy.sparse as sp


BIOCHEM_MASSES = {
    "self": 0.0,
    "C": 12.0,
    "H": 1.00783,
    "H2": 2.01566,
    "O": 15.99491,
    "NH2-O": 0.02381,
    "PO3": 78.95851,
    "S": 31.97207,
}


class Spectra:
    @staticmethod
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

    def __init__(
        self,
        mzis,
        pmzs=None,
        intensity_power=0.5,
        fragment="mz",
        measure="i",
        tolerance=0.01,
        mass_shifts=[],
        blend_steps=1,
        mirror_masses=True,
        trim_empty=False,
        metadata=None,
    ):
        assert (pmzs is None) or (len(mzis) == len(pmzs))

        if trim_empty:
            kept, mzis = np.array(
                [[idx, mzi] for idx, mzi in enumerate(mzis) if mzi.size > 0],
                dtype=object,
            ).T
        else:
            kept = np.arange(len(mzis))

        if pmzs is not None:
            pmzs = np.asarray(pmzs)[kept]

        ids = np.concatenate([[i] * mzi.shape[1] for i, mzi in enumerate(mzis)])
        mzis = np.concatenate(mzis, axis=1)

        self._id = ids
        self._blanks = np.setdiff1d(np.arange(ids[-1] + 1), kept)

        self._mz = mzis[0]
        self._pmz = pmzs
        self._nl = None

        self._i = mzis[1] ** intensity_power
        self._intensity_power = intensity_power
        self._c = None

        self._tolerance = tolerance
        self._mass_shifts = np.asarray(mass_shifts)
        self._blend_steps = blend_steps
        self._mirror_masses = mirror_masses

        self._kernel = None
        self._norm = None

        self.metadata = metadata

        if fragment == "mz":
            self._fragment = self._mz
        elif fragment == "nl":
            self._fragment = self._nl_proxy

        if measure == "i":
            self._measure = self._i
        elif measure == "c":
            self._measure = self._c_proxy

        self._update()

    #############
    # Properties
    #############

    @property
    def _nl_proxy(self):
        if self._nl is None:
            self._nl = self._pmz[self._id] - self._mz
        return self._nl

    @property
    def _c_proxy(self):
        if self._c is None:
            self._c = self._i.astype(bool)
        return self._c

    @property
    def fragment(self):
        if self._fragment is self._mz:
            return "mz"
        elif self._fragment is self._nl:
            return "nl"

    @fragment.setter
    def fragment(self, new_fragment):
        if (new_fragment == "mz") and (self._fragment is self._nl):
            self._fragment = self._mz
            self._kernel = -(self._kernel - self._pmz[self._id])
        elif (new_fragment == "nl") and (self._fragment is self._mz):
            self._fragment = self._nl_proxy
            self._kernel = self._pmz[self._id] - self._kernel

    @property
    def measure(self):
        if self._measure is self._i:
            return "i"
        elif self._measure is self._c:
            return "c"

    @measure.setter
    def measure(self, new_measure):
        if (new_measure == "i") and (self._measure is self._c):
            self._measure = self._i
            self._normalize()
        elif (new_measure == "c") and (self._measure is self._i):
            self._measure = self.c
            self._normalize()

    @property
    def intensity_power(self):
        return self._intensity_power

    @intensity_power.setter
    def intensity_power(self, new_intensity_power):
        if self._intensity_power != new_intensity_power:
            self._i = self._i ** (new_intensity_power / self._intensity_power)
            self._intensity_power = new_intensity_power
            self._normalize()

    @property
    def tolerance(self):
        return self._tolerance

    @tolerance.setter
    def tolerance(self, new_tolerance):
        if self._tolerance != new_tolerance:
            self._tolerance = new_tolerance
            self._normalize()

    @property
    def mass_shifts(self):
        return self._mass_shifts

    @mass_shifts.setter
    def mass_shifts(self, new_mass_shifts):
        if (self._mass_shifts.shape != new_mass_shifts.shape) or (
            self._mass_shifts != new_mass_shifts
        ).any():
            self._mass_shifts = new_mass_shifts
            self._update()

    @property
    def blend_steps(self):
        return self._blend_steps

    @blend_steps.setter
    def blend_steps(self, new_blend_steps):
        if self._blend_steps != new_blend_steps:
            self._blend_steps = new_blend_steps
            self._update()

    @property
    def mirror_masses(self):
        return self._mirror_masses

    @mirror_masses.setter
    def mirror_masses(self, mirror_masses):
        if self._mirror_masses != new_mirror_masses:
            self._mirror_masses = new_mirror_masses
            self._update()

    #################
    # Helper Methods
    #################

    def _update(self):
        if self._mirror_masses:
            mass_shifts = np.multiply.outer(self._mass_shifts, [1, -1]).flatten()
        else:
            mass_shifts = self._mass_shifts

        mass_shifts = np.unique(np.append(0.0, mass_shifts))

        # Recursively "blend" mass_shifts within a specified number of steps
        def blend(mass_shifts, blend_steps):
            if blend_steps == 0:
                return np.array([0.0])
            if blend_steps == 1:
                return mass_shifts
            else:
                return np.add.outer(mass_shifts, blend(mass_shifts, blend_steps - 1))

        mass_shifts = np.unique(blend(mass_shifts, self._blend_steps).flatten())

        self._kernel = np.add.outer(mass_shifts, self._fragment)

        self._normalize()

    def _normalize(self):
        interlace = self._interlace(self, same_id=True)

        interlace = interlace[:, np.argsort(self._id[interlace[0]])]

        same_id = self._id[interlace[0]] == self._id[interlace[1]]

        id_mask = self._id[interlace[0]][1:] != self._id[interlace[0]][:-1]
        id_mask = np.append(True, id_mask)
        (id_edge,) = np.nonzero(id_mask)

        self._norm = (
            np.add.reduceat(
                same_id
                * self._parabolic_blur(self, interlace)
                * self._measure[interlace[0]]
                * self._measure[interlace[1]],
                id_edge,
            )
            ** -0.5
        )

        if self._measure is self._c:
            self._norm *= np.add.reduceat(self._c[interlace[0]], id_edge) ** 0.5

    def _interlace(self, other, same_id=False):
        assert self._tolerance == other.tolerance

        self_fragment = self._fragment
        other_kernel = other._kernel
        tolerance = self._tolerance

        if same_id:
            assert self is other

            self_fragment = 1j * self_fragment + self._id
            other_kernel = 1j * other_kernel + self._id
            tolerance = 1j * tolerance

        sort_idx = np.argsort(self_fragment)

        overlap = np.array(
            [
                np.searchsorted(
                    self_fragment[sort_idx],
                    other_kernel.ravel() - tolerance,
                    "left",
                ),
                np.searchsorted(
                    self_fragment[sort_idx],
                    other_kernel.ravel() + tolerance,
                    "right",
                ),
            ]
        )

        interlace_idx = np.repeat(
            np.arange(overlap.shape[1]), overlap[1] - overlap[0], axis=0
        )

        interlace = np.array(
            [
                sort_idx[self._multi_arange(overlap[:, overlap[0] != overlap[1]].T)],
                interlace_idx % other_kernel.shape[1],
                interlace_idx // other_kernel.shape[1],
            ]
        )

        return interlace

    def _parabolic_blur(self, other, interlace):
        mz_diff = (
            self._fragment[interlace[0]] - other._kernel[interlace[2], interlace[1]]
        )

        return (3 / 4) * (1 - abs(mz_diff / self._tolerance) ** 2)

    #################
    # Public Methods
    #################

    def score(self, other):
        assert self.fragment == other.fragment
        assert self.measure == other.measure

        assert self.tolerance == other.tolerance
        assert (self._mass_shifts.shape == other._mass_shifts.shape) and (
            self._mass_shifts == other._mass_shifts
        ).all()
        assert self.blend_steps == other.blend_steps
        assert self.mirror_masses == other.mirror_masses

        ordered = self._fragment.size <= other._fragment.size
        if not ordered:
            self, other = other, self

        interlace = self._interlace(other)
        mz_bins = np.unique(interlace[0])

        S = sp.coo_matrix(
            (
                (self._norm[self._id] * self._measure)[mz_bins],
                (self._id[mz_bins], mz_bins),
            ),
        ).tocsr()

        O = sp.coo_matrix(
            (
                self._parabolic_blur(other, interlace)
                * (other._norm[other._id] * other._measure)[interlace[1]],
                (other._id[interlace[1]], interlace[0]),
            ),
        ).T.tocsc()

        blink_score = S.dot(O)
        if not ordered:
            blink_score = blink_score.T

        return blink_score
