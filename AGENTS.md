# AGENTS.md: cudals

Orientation for an AI agent (or a new human) working on `cudals`. This file is authoritative
for conventions in this repo.

## What cudals is

A jaxls-shaped nonlinear least-squares library whose only solver is CUDA. `analyze()` traces a
problem's residuals to a jaxpr, emits them (and their forward-mode Jacobians) as straight-line
device code in `cost_gen.cuh`, and compiles that header into a fused LM / augmented-Lagrangian
kernel. `solve()` / `solve_batch()` is one FFI launch. **There is no JAX fallback.** An
unsupported problem raises.

**cudals is independent.** It never imports pyroffi, GRiM or GRiD, and it ships its own
solver kernels and its own copy of GLASS. Host packages extend it through two public hooks:

- `register_baked_type(cls, resolve)`: instances of `cls` are baked into the build, and
  `resolve` maps a traced instance to its concrete original.
- `NativeResidual(..., include_dirs=...)`: hand-written device rows, with the directories their
  source includes from.

Keep it that way. If cudals needs something from a host package, add a hook here and have the
host register with it.

## Layout

Source is in `src/cudals/`.

- `_problem.py`: the jaxls API (`Var`, `VarValues`, `Cost`, `LeastSquaresProblem`, configs,
  Lie-group variables).
- `_compile.py`: planning (baked vs runtime arguments), tracing, the dense and banded header
  generators, tier choice, the header cache and `AnalyzedLeastSquaresProblem`.
- `_emit.py`: jaxpr → CUDA. Primitive handlers, folding, cooperative loops.
- `_native.py`: `NativeResidual` and the host-side data upload.
- `_build.py`: `nvcc` builds of the solver kernels around a header, the `.so` cache, FFI
  registration. Its docstring is the header contract.
- `kernels/cost_lm.cu` (dense, thread / warp / block tiers), `kernels/cost_lm_banded.cu`
  (banded, one block per problem), `kernels/tier_solve.cuh` (three-tier Cholesky).
- `kernels/glass/`: vendored GLASS, upstream commit `9a24f378`. Don't edit it. Re-copy it from
  upstream and update the commit in `README.md`.

## Test

```bash
CUDA_VISIBLE_DEVICES=<free idx> python -m pytest tests -q
```

- `tests/test_kernels.py`: the solver kernels on hand-written headers. Dense: a
  constrained Rosenbrock at every tier and precision against the oracle's local minimizers.
  Banded: a linear chain against numpy's exact least squares. Both are run twice and must be
  bit-identical.
- PyRoFFI's `tests/test_costs.py` covers the emitter (every primitive handler and fold rule
  against JAX), the compiler, the caches and compiled robot solves. Run it after changing
  `_compile.py` or `_emit.py`.

## Rules

1. **Match jaxls.** Names, signatures and semantics follow jaxls. Intentional differences go
   in `README.md`'s "Behaviour that intentionally differs" list.
2. **No silent fallbacks, no hot-path guards.** Unsupported input raises at `analyze()`. Don't
   add runtime checks to the kernels or the solve loop.
3. **Validate generated code numerically.** ptxas has silently miscompiled large straight-line
   traces. A new primitive handler gets an emitter test against JAX. A kernel change gets the
   kernel tests and a compiled-solve test.
4. **Deterministic and tier-invariant.** A kernel's output must be bit-identical run to run and
   must not depend on a problem's batch slot. No atomics decide an output. Tiers may differ
   only by Cholesky round-off.
5. **One problem never spans blocks** in the dense solver. The banded solver's per-problem
   control and factorization stay in one block.
6. **Cache keys must cover everything that changes the output:** header, flags, kernel and
   GLASS sources, native include directories, and the source of packages defining baked types.
   A stale cache hit is a wrong answer.
