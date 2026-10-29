"""nvcc builds of the LM / augmented-Lagrangian solvers around a generated problem header.

The solvers (``kernels/cost_lm.cu``, ``kernels/cost_lm_banded.cu``) are problem-agnostic: the
compiler (:mod:`._compile`) writes a problem's residuals as device code in ``cost_gen.cuh``,
and this module compiles the two together and registers the result as a JAX FFI target.
The header contracts:

``cost_gen.cuh`` (dense, :func:`dense_solver`), in ``namespace cudals::costs_gen``:
    ``n_x``, ``n_r``, ``row_kind[n_r]`` (0 cost, 1 equality == 0, 2 inequality <= 0),
    ``has_constraints``, ``scratch_size`` and
    ``residual(x, p, r, scratch, rank, size)``,
    ``residual_jacobian(x, p, r, J, scratch, rank, size)`` (J row-major ``(n_r, n_x)``),
    written in GLASS form: lanes run straight-line code redundantly and split loops
    ``for (i = rank; i < n; i += size)`` each followed by ``CUDALS_COST_SYNC()``.
    Optional variants, enabled by the header defining the macro: ``CUDALS_COST_JVP``
    (``residual_jvp``, column-parallel tiers), ``CUDALS_COST_NORMAL_EQUATIONS``
    (``normal_equations``, symbolic H/g), ``CUDALS_COST_SPARSE_NE`` (``g_ptr``/``g_rows``,
    ``h_ptr``/``h_rows``/``h_a``/``h_c``/``h_nnz``: the structural H pattern),
    ``CUDALS_COST_MANIFOLD`` (Lie-group variables: ``n_q`` parameters stored per problem,
    the solve in the ``n_x``-dimensional tangent space, and ``retract(x, p, t, xn)`` the
    step; ``x0`` and ``x`` are then ``(B, n_q)``).

``cost_gen.cuh`` (banded, :func:`banded_solver`): cost GROUPS of identical instances over
    local variables -- ``stage_res(g, xl, pl, r)``, ``stage_jvp(g, xl, pl, t, r, jt)``,
    the index tables ``vtab``/``ptab`` and the group layout (``n_groups``, ``g_nloc``,
    ``g_nploc``, ``g_nr``, ``g_vtab0``, ``g_ptab0``, ``g_row0``, ``g_j0``,
    ``g_inst_prefix``, ``g_col_prefix``, ``n_inst_total``, ``n_col_total``, ``max_nloc``,
    ``max_nploc``, ``max_nr``, ``j_size``), ``band``, ``fixed[n_x]``, ``n_x``, ``n_r``,
    ``row_kind``, ``has_constraints``.

Every solver's FFI target takes ``x0 (B, n_x)``, ``params (B | 1, max(n_p, 1))`` and returns
``x``, ``cost``, ``viol`` and a byte workspace; settings are attributes (see the kernels).

Builds are cached on disk (``$CUDALS_SO_CACHE``, else ``~/.cache/cudals/so``) under a key
over the kernel sources, GLASS, the generated header, any extra include directories and the
flags, so each structure compiles once; concurrent builds of the same key are safe.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

KERNELS_DIR = Path(__file__).parent / "kernels"
GLASS_DIR = KERNELS_DIR / "glass"
TIERS = {"thread": 0, "warp": 1, "block": 2}
LIBRARIES: dict = {}   # FFI target name -> its loaded library (for host-side setters)


def _cache_root() -> Path:
    env = os.environ.get("CUDALS_SO_CACHE")
    if env:
        return Path(env)
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "cudals" / "so"


def _compile(kernel: str, header: str, flags: tuple[str, ...], include_dirs: tuple[str, ...]) -> Path:
    """nvcc ``kernels/<kernel>.cu`` with ``cost_gen.cuh`` = ``header`` next to it."""
    import jaxlib
    xla_include = Path(jaxlib.__file__).parent / "include"
    flags = ("-O3", "-std=c++17", os.environ.get("CUDALS_ARCH", "-arch=native"), "--shared",
             "--compiler-options", "-fPIC", *flags)
    dirs = [KERNELS_DIR, GLASS_DIR, *map(Path, include_dirs)]
    inputs = sorted({p for d in dirs for p in d.rglob("*.cu*")})
    key = hashlib.sha1("\x00".join(
        [kernel, header, " ".join(flags), str(xla_include), *map(str, include_dirs),
         *(f"{p}\x01{p.read_text()}" for p in inputs)]).encode()).hexdigest()
    out_dir = _cache_root() / key
    so = out_dir / f"{kernel}.so"
    if so.is_file():
        return so
    _cache_root().mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{key}.", dir=_cache_root()))
    try:
        (tmp / "cost_gen.cuh").write_text(header)
        cmd = ["nvcc", *flags, f"-I{tmp}", *(f"-I{d}" for d in dirs), f"-I{xla_include}",
               "-o", str(tmp / so.name), str(KERNELS_DIR / f"{kernel}.cu")]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            lines = result.stderr.splitlines()
            errors = [ln for i, ln in enumerate(lines)
                      if "error" in ln or any("error" in x for x in lines[max(i - 2, 0):i])]
            raise RuntimeError(f"nvcc failed to build {kernel}:\n" + "\n".join(errors or lines[-40:]))
        try:
            tmp.rename(out_dir)
        except OSError:   # another process finished the same build first
            pass
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return so


def _register(so: Path) -> str:
    """Register the library's ``CostLmFfi`` as a CUDA custom-call target; returns its name."""
    import jax
    lib = ctypes.CDLL(str(so))
    capsule_new = ctypes.pythonapi.PyCapsule_New
    capsule_new.restype = ctypes.py_object
    capsule_new.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    name = f"cudals_CostLmFfi_{so.parent.name[:16]}"
    if name not in LIBRARIES:
        capsule = capsule_new(ctypes.cast(lib.CostLmFfi, ctypes.c_void_p), b"xla._CUSTOM_CALL_TARGET", None)
        jax.ffi.register_ffi_target(name, capsule, platform="CUDA")
        LIBRARIES[name] = lib
    return name


def _precision_flags(real: str, io: str) -> tuple[str, ...]:
    return (f"-DCUDALS_COST_REAL={real}", f"-DCUDALS_COST_IO={io}",
            *(("-DCUDALS_COST_IO_F64",) if io == "double" else ()))


@lru_cache(maxsize=None)
def dense_solver(header: str, tier: str, real: str = "float", io: str = "float",
                 include_dirs: tuple[str, ...] = ()) -> str:
    """LM / augmented Lagrangian at one tier (``kernels/cost_lm.cu``) -> FFI target."""
    so = _compile("cost_lm", header, (f"-DCUDALS_COST_TIER={TIERS[tier]}", *_precision_flags(real, io)),
                  include_dirs)
    return _register(so)


@lru_cache(maxsize=None)
def banded_solver(header: str, real: str = "double", io: str = "double",
                  include_dirs: tuple[str, ...] = (), profile: bool = False) -> tuple[str, int]:
    """LM / augmented Lagrangian with a banded Cholesky (``kernels/cost_lm_banded.cu``)
    -> (FFI target, workspace bytes per problem). Host-driven iterations: the Jacobian,
    assembly and trial residual spread over many blocks per problem, the factorization and
    control run in one block per problem."""
    so = _compile("cost_lm_banded", header,
                  (*_precision_flags(real, io), *(("-DCUDALS_COST_PROFILE",) if profile else ())),
                  include_dirs)
    lib = ctypes.CDLL(str(so))
    lib.cudals_cost_lm_banded_workspace_bytes.restype = ctypes.c_size_t
    return _register(so), int(lib.cudals_cost_lm_banded_workspace_bytes())
