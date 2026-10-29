"""Native residuals: costs whose rows and Jacobian come from hand-written device code.

The cost compiler traces a residual's jaxpr and emits it as straight-line CUDA, which is the
right form for kinematic chains but explodes for residuals that are naturally loops: a
collision residual is (spheres x obstacles) distances and their derivatives, and its fully
unrolled body grows to megabytes that ptxas takes minutes on. A ``NativeResidual`` instead
supplies one ``__device__`` function per cost instance; the traced residual only reserves its
rows (zeros), so row kinds, weights and constraint handling are unchanged, and the generated
header calls the function after the traced body to fill those rows of ``r`` and ``J`` (or
``jt = J t`` in the column-parallel variant).

Device function contract (one cost instance, one variable of ``dof`` coordinates)::

    __device__ void <fn>(const float* q, float* r, float* J)   // J: (rows, dof) or nullptr
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


@dataclass(frozen=True)
class NativeResidual:
    fn: str          # device function name
    source: str      # its CUDA definition (and the tables it reads)
    rows: int        # residual rows per instance
    dof: int         # coordinates of the instance's variable
    # Runtime data the device code reads from mutable __device__ buffers, uploaded before every
    # solve by the exported host function ``setter`` (so the data is not part of the source, and
    # the compiled kernel is shared by every problem whose data differs only in value).
    setter: str = ""
    data: tuple = field(default=(), compare=False, hash=False)
    # Directories the source #includes from (part of the build key).
    include_dirs: tuple[str, ...] = ()

    @property
    def digest(self) -> str:
        return hashlib.sha1(self.source.encode()).hexdigest()[:16]


def upload(native: NativeResidual, lib) -> None:
    """Write a native residual's runtime data into the library that runs it."""
    import ctypes
    if not native.setter:
        return
    ptr = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    args = []
    for a in native.data:
        args += [ptr(a), ctypes.c_int(len(a))]
    err = getattr(lib, native.setter)(*args)
    if err != 0:
        raise RuntimeError(f"{native.setter} failed ({err})")
