# `cudals`: jaxls-shaped Nonlinear Least Squares, Compiled to One CUDA Kernel

`cudals` solves batched nonlinear least-squares problems, with equality and inequality
constraints, on NVIDIA GPUs. You write variables and costs as in
[jaxls](https://github.com/brentyi/jaxls). `analyze()` traces the residuals and their exact
Jacobians into straight-line CUDA, compiles them into a fused Levenberg-Marquardt /
augmented-Lagrangian solver, and `solve()` / `solve_batch()` runs as a single kernel launch
with no JAX in the loop.

- Dense solver at three tiers: one problem per thread, per warp, or per block
- Banded solver for trajectory-structured problems (costs over neighbouring knots)
- Constraints `== 0`, `<= 0`, `>= 0` by augmented Lagrangian
- Euclidean and Lie-group variables (`SE3Var`, `SO3Var`, `SE2Var`, `SO2Var`)
- Runtime inputs (`jax.Array` arguments) reuse the build; builds are cached on disk
- Composes with `jax.jit` and `jax.vmap` as an FFI custom call

There is no JAX fallback. A problem the compiler cannot handle, or a machine without a CUDA
device and `nvcc`, raises.

`cudals` is the solver behind [PyRoFFI](https://github.com/CoMMALab/pyroffi)'s `pk.LeastSquaresProblem`, but it
does not depend on PyRoFFI or GRiM. It needs JAX, jaxlie, NumPy and `nvcc`.

## Coming from jaxls

Swap the import:

```python
import cudals as jaxls  # was: import jaxls
```

```python
import jax
import jax.numpy as jnp
import cudals as ls


class X(ls.Var[jax.Array], default_factory=lambda: jnp.zeros(2)): ...


@ls.Cost.factory
def objective(vals, x):
    return vals[x]


@ls.Cost.factory(kind="constraint_eq_zero")
def on_line(vals, x):
    return vals[x][:1] + vals[x][1:] - 1.0


x = X(0)
sol = ls.LeastSquaresProblem([objective(x), on_line(x)], [x]).analyze().solve()
sol[x]  # [0.5, 0.5]
```

Behaviour that intentionally differs from jaxls:

- `analyze()` compiles to CUDA. The first build of a structure runs `nvcc` (seconds for
  small problems, minutes for long trajectories). After that it is a cache hit.
- `linear_solver`, `sparse_mode`, `ConjugateGradientConfig` and most trust-region and
  augmented-Lagrangian options are accepted and ignored. The fields the kernel reads are
  marked `# used` in `_problem.py`.
- `LeastSquaresProblem(..., fixed=[...])` holds variables at their initial value (an
  extension, solved by the banded solver).

## Batching

The kernels pay off at large batches. Build the problem once, then solve many instances in
one launch:

```python
analyzed = problem.analyze()
params = jax.vmap(lambda t: build_problem(t).analyze().params())(targets)  # (B, n_p)
x, cost, violation = analyzed.solve_batch(params, x0)                      # x0: (B, n_q)
```

`tier="auto"` picks thread, warp or block from a batch-size heuristic, or from the table that
`analyzed.calibrate()` measures on your GPU.

## Baked constants vs runtime inputs

- **Runtime inputs:** every `jax.Array` argument of a cost (targets, rest poses, obstacle
  geometry). A new value reuses the build.
- **Baked into the build:** the problem's structure, NumPy arrays, Python scalars, and
  instances of types registered with `cudals.register_baked_type` (PyRoFFI registers its robot
  and collision models). A new value is a new build.

To re-solve with new data without a rebuild, pass `jax.Array`s.

## Constraints and warm starts

Constraints are solved by an augmented Lagrangian and land on a local constrained optimum (a
KKT point). Seed a constrained problem with a good unconstrained solution through
`initial_vals`, exactly as in jaxls:

```python
sol = problem.analyze().solve(
    initial_vals=ls.VarValues.make([x.with_value(x_warm)]),
    termination=ls.TerminationConfig(max_iterations=300),
    augmented_lagrangian=ls.AugmentedLagrangianConfig(max_iterations=50))
```

`max_iterations` counts inner steps across all AL updates, as in jaxls, so the default of 100
can stop before a constrained problem is feasible. Check `SolveSummary.constraint_violation`
(`solve(..., return_summary=True)`).

## Native residuals

A residual that is naturally a loop (a collision sum over spheres and obstacles) unrolls into
megabytes of straight-line code. A `cudals.NativeResidual` supplies hand-written device code
for those rows instead. The traced residual only reserves them, and constraint handling is
unchanged. See `_native.py` for the device-function contract.

## Limits

- At most 32 tangent dimensions at the thread and warp tiers, 128 at the block tier. Larger
  problems with trajectory structure use the banded solver.
- Residuals must have static shapes and no data-dependent loops.
- One batch axis per cost.

## Environment variables

| Variable | Default | Effect |
| --- | --- | --- |
| `CUDALS_CACHE` | `~/.cache/cudals` | Generated-header cache. `0` disables it. (`PYROFFI_COST_CACHE` is still honoured.) |
| `CUDALS_SO_CACHE` | `~/.cache/cudals/so` | Compiled-library cache |
| `CUDALS_ARCH` | `-arch=native` | `nvcc` architecture flag |

## Third-party code

`kernels/glass/` is a vendored copy of [GLASS](https://github.com/A2R-Lab/GLASS) (A²R Lab,
MIT, upstream commit `9a24f378`), the thread / warp / block linear-algebra headers behind the
Cholesky solves. `kernels/tier_solve.cuh` is derived from GRiM's `glass_solve.cuh` (MIT).
