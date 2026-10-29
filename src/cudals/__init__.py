"""cudals: jaxls-shaped nonlinear least squares, compiled to one CUDA kernel.

Write variables and costs as in jaxls (``cudals.Var``, ``cudals.Cost.factory``,
``cudals.LeastSquaresProblem``); ``analyze()`` traces the residuals and their Jacobians into
straight-line CUDA around a Levenberg-Marquardt / augmented-Lagrangian solver, and
``solve()`` / ``solve_batch()`` is one launch with no JAX in the loop. There is no JAX
fallback: a problem that cannot be compiled raises.

Model types that should be baked into a build (a robot, a collision model) are registered
with :func:`register_baked_type`; pyroffi does this for its own.
"""

from ._compile import TIERS as TIERS
from ._compile import AnalyzedLeastSquaresProblem as AnalyzedLeastSquaresProblem
from ._compile import register_baked_type as register_baked_type
from ._native import NativeResidual as NativeResidual
from ._problem import AugmentedLagrangianConfig as AugmentedLagrangianConfig
from ._problem import ConjugateGradientConfig as ConjugateGradientConfig
from ._problem import Cost as Cost
from ._problem import LeastSquaresProblem as LeastSquaresProblem
from ._problem import SE2Var as SE2Var
from ._problem import SE3Var as SE3Var
from ._problem import SO2Var as SO2Var
from ._problem import SO3Var as SO3Var
from ._problem import SolveSummary as SolveSummary
from ._problem import TerminationConfig as TerminationConfig
from ._problem import TrustRegionConfig as TrustRegionConfig
from ._problem import Var as Var
from ._problem import VarValues as VarValues
from ._problem import VarWithValue as VarWithValue
