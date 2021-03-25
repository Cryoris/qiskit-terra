"""A generalized SPSA optimizer including support for Hessians."""

from typing import Iterator, Optional, Union, Callable

import numpy as np
from qiskit.circuit import ParameterVector
from qiskit.opflow import StateFn, CircuitSampler

from ._spsa import SPSA

# the overlap function
OVERLAP = Callable[[np.ndarray, np.ndarray], float]

# parameters, loss, stepsize, number of function evaluations, accepted
CALLBACK = Callable[[np.ndarray, float, float, int, bool], None]


class QNSPSA(SPSA):
    """Quantum Natural SPSA."""

    def __init__(self,
                 overlap_fn: OVERLAP,
                 maxiter: int = 100,
                 blocking: bool = False,
                 learning_rate: Optional[Union[float, Callable[[], Iterator]]] = None,
                 perturbation: Optional[Union[float, Callable[[], Iterator]]] = None,
                 tolerance: float = 1e-7,
                 callback: Optional[CALLBACK] = None,
                 # 2-SPSA arguments
                 hessian_delay: int = 0,
                 hessian_resamplings: int = 1,
                 lse_solver: Optional[Union[str,
                                            Callable[[np.ndarray, np.ndarray], np.ndarray]]] = None,
                 regularization: Optional[float] = None,
                 perturbation_dims: Optional[int] = None,
                 initial_hessian: Optional[np.ndarray] = None,
                 ) -> None:
        r"""
        Args:
            maxiter: The maximum number of iterations.
            blocking: If True, only accepts updates that improve the loss.
            learning_rate: A generator yielding learning rates for the parameter updates,
                :math:`a_k`.
            perturbation: A generator yielding the perturbation magnitudes :math:`c_k`.
            tolerance: If the norm of the parameter update is smaller than this threshold, the
                optimizer is converged.
            callback: A callback function passed information in each iteration step. The
                information is, in this order: the parameters, the function value, the number
                of function evaluations, the stepsize, whether the step was accepted.
            second_order: If True, use 2-SPSA instead of SPSA. In 2-SPSA, the Hessian is estimated
                additionally to the gradient, and the gradient is preconditioned with the inverse
                of the Hessian to improve convergence.
            hessian_delay: Start preconditioning only after a certain number of iterations.
                Can be useful to first get a stable average over the last iterations before using
                the preconditioner.
            hessian_resamplings: In each step, sample the preconditioner this many times. Default
                is 1.
            lse_solver: The method to solve for the inverse of the preconditioner. Per default an
                exact LSE solver is used, but can e.g. be overwritten by a minimization routine.
            regularization: To ensure the preconditioner is symmetric and positive definite, the
                identity times a small coefficient is added to it. This generator yields that
                coefficient.
            perturbation_dims: The number of dimensions to perturb at once. Per default all
                dimensions are perturbed simulatneously.
            initial_hessian: The initial guess for the Hessian. By default the identity matrix
                is used.
        """
        super().__init__(maxiter,
                         blocking,
                         trust_region=False,
                         learning_rate=learning_rate,
                         perturbation=perturbation,
                         tolerance=tolerance,
                         callback=callback,
                         second_order=True,
                         hessian_delay=hessian_delay,
                         hessian_resamplings=hessian_resamplings,
                         lse_solver=lse_solver,
                         regularization=regularization,
                         perturbation_dims=perturbation_dims,
                         initial_hessian=initial_hessian)

        self.overlap_fn = overlap_fn

    # pylint: disable=unused-argument
    def _point_estimate(self, loss, x, eps, delta1, delta2, plus, minus):
        pert1, pert2 = eps * delta1, eps * delta2

        plus = self.overlap_fn(x, x + pert1)
        minus = self.overlap_fn(x, x - pert1)

        # compute the preconditioner point estimate
        diff = self.overlap_fn(x, x + pert1 + pert2) - plus
        diff -= self.overlap_fn(x, x - pert1 + pert2) - minus
        diff /= 2 * eps ** 2

        self._nfev += 4

        rank_one = np.outer(delta1, delta2)
        estimate = diff * (rank_one + rank_one.T) / 2

        return estimate

    @staticmethod
    def get_overlap(circuit, backend=None, expectation=None):
        """Get the overlap function."""
        params_x = ParameterVector('x', circuit.num_parameters)
        params_y = ParameterVector('y', circuit.num_parameters)

        expression = ~StateFn(circuit.assign_parameters(
            params_x)) @ StateFn(circuit.assign_parameters(params_y))

        if expectation is not None:
            expression = expectation.convert(expression)

        if backend is None:
            def overlap_fn(values_x, values_y):
                value_dict = dict(zip(params_x[:] + params_y[:],
                                      values_x.tolist() + values_y.tolist()))
                return -0.5 * np.abs(expression.bind_parameters(value_dict).eval()) ** 2
        else:
            sampler = CircuitSampler(backend)

            def overlap_fn(values_x, values_y):
                value_dict = dict(zip(params_x[:] + params_y[:],
                                      values_x.tolist() + values_y.tolist()))
                return -0.5 * np.abs(sampler.convert(expression, params=value_dict).eval()) ** 2

        return overlap_fn
