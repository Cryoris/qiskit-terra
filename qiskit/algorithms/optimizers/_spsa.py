"""A generalized SPSA optimizer including support for Hessians."""

from typing import Iterator, Optional, Union, Callable, Tuple
import logging
import warnings
from time import time

import scipy
import numpy as np

from qiskit.algorithms.optimizers import Optimizer, OptimizerSupportLevel

# a preconditioner can either be a function (e.g. loss function to obtain the Hessian)
# or a metric (e.g. Fubini-Study metric to obtain the quantum Fisher information)
PRECONDITIONER = Union[Callable[[float], float],
                       Callable[[float, float], float]]

# parameters, loss, stepsize, number of function evaluations, accepted
CALLBACK = Callable[[np.ndarray, float, float, int, bool], None]

logger = logging.getLogger(__name__)


class SPSA(Optimizer):
    """A generalized SPSA optimizer including support for Hessians."""

    def __init__(self, maxiter: int = 100,
                 blocking: bool = False,  # save_steps: int = 1,
                 trust_region: Union[bool, str] = False,  # last_avg: int = 1,
                 # c0: float = _C0,
                 learning_rate: Optional[Union[float, Callable[[], Iterator]]] = None,
                 # c1: float = 0.1,
                 perturbation: Optional[Union[float, Callable[[], Iterator]]] = None,
                 tolerance: float = 1e-7,  # c2: float = 0.602,
                 callback: Optional[CALLBACK] = None,  # c3: float = 0.101,
                 # 2-SPSA arguments
                 second_order: bool = False,  # c4: float = 0,
                 hessian_delay: int = 0,  # skip_calibration: bool = False) -> None:
                 hessian_resamplings: int = 1,
                 lse_solver: Optional[Union[str,
                                            Callable[[np.ndarray, np.ndarray], np.ndarray]]] = None,
                 regularization: Optional[float] = None,
                 perturbation_dims: Optional[int] = None,
                 initial_hessian: Optional[np.ndarray] = None,
                 # TODO handle deprecated arguments?
                 #  *,  # swallow accidential additional unnamed args to not set deprecated args
                 #  save_steps: Optional[int] = None,
                 #  last_avg: Optional[int] = None,
                 #  c0: Optional[float] = None,
                 #  c1: Optional[float] = None,
                 #  c2: Optional[float] = None,
                 #  c3: Optional[float] = None,
                 #  c4: Optional[float] = None,
                 #  skip_calibration: Optional[bool] = None
                 ) -> None:
        r"""
        Args:
            maxiter: The maximum number of iterations.
            blocking: If True, only accepts updates that improve the loss.
            trust_region: If True, restricts norm of the random direction to be <= 1.
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
        super().__init__()

        if regularization is None:
            regularization = 0.01

        if isinstance(learning_rate, float):
            self.learning_rate = lambda: constant(learning_rate)
        else:
            self.learning_rate = learning_rate

        if isinstance(perturbation, float):
            self.perturbation = lambda: constant(perturbation)
        else:
            self.perturbation = perturbation

        if lse_solver is None:
            lse_solver = np.linalg.solve

        self.maxiter = maxiter
        self.blocking = blocking
        self.trust_region = trust_region
        self.callback = callback
        self.second_order = second_order  # more logic included in the setter
        self.tolerance = tolerance
        self.hessian_delay = hessian_delay
        self.hessian_resamplings = hessian_resamplings
        self.lse_solver = lse_solver
        self.regularization = regularization
        self.perturbation_dims = perturbation_dims
        self.initial_hessian = initial_hessian
        self.trust_region = trust_region

        # runtime arguments
        self._nfev = None
        self._moving_avg = None  # moving average of the preconditioner

    @staticmethod
    def calibrate(loss: Callable[[np.ndarray], float],
                  initial_point: np.ndarray,
                  c: float = 0.2,
                  stability_constant: float = 0,
                  target_magnitude: Optional[float] = None,  # 2 pi / 10
                  alpha: float = 0.602,
                  gamma: float = 0.101,
                  modelspace: bool = False) -> Tuple[Iterator[float], Iterator[float]]:
        r"""Calibrate SPSA parameters with a powerseries as learning rate and perturbation coeffs.

        The powerseries are:

        .. math::

            a_k = \frac{a}{(A + k + 1)^\alpha}, c_k = \frac{c}{(k + 1)^\gamma}

        Args:
            loss: The loss function.
            initial_point: The initial guess of the iteration.
            c: The initial perturbation magnitude.
            stability_constant: The value of `A`.
            target_magnitude: The target magnitude for the first update step.
            alpha: The exponent of the learning rate powerseries.
            gamma: The exponent of the perturbation powerseries.
            modelspace: Whether the target magnitude is the difference of parameter values
                or function values (= model space).

        Returns:
            tuple(generator, generator): A tuple of powerseries generators, the first one for the
                learning rate and the second one for the perturbation.
        """
        if target_magnitude is None:
            target_magnitude = 2 * np.pi / 10

        dim = len(initial_point)

        # compute the average magnitude of the first step
        steps = 25
        avg_magnitudes = 0
        for _ in range(steps):
            # compute the random directon
            pert = np.array([1 - 2 * np.random.binomial(1, 0.5)
                             for _ in range(dim)])
            delta = loss(initial_point + c * pert) - \
                loss(initial_point - c * pert)
            avg_magnitudes += np.abs(delta / (2 * c))

        avg_magnitudes /= steps

        if modelspace:
            a = target_magnitude / (avg_magnitudes ** 2)
        else:
            a = target_magnitude / avg_magnitudes

        # compute the rescaling factor for correct first learning rate
        if a < 1e-10:
            warnings.warn(f'Calibration failed, using {target_magnitude} for `a`')
            a = target_magnitude

        # set up the powerseries
        def learning_rate():
            return powerseries(a, alpha, stability_constant)

        def perturbation():
            return powerseries(c, gamma)

        return learning_rate, perturbation

    def _compute_gradient(self, loss, x, eps, delta):
        # compute the gradient approximation and additionally return the loss function evaluations
        plus, minus = loss(x + eps * delta), loss(x - eps * delta)
        self._nfev += 2
        return (plus - minus) / (2 * eps) * delta, plus, minus

    def _point_estimate(self, loss, x, eps, delta1, delta2, plus, minus):
        pert1, pert2 = eps * delta1, eps * delta2

        # if the loss is the preconditioner we can save two evaluations
        # else:
        #     plus = self._eval_preconditioner(x, pert1)
        #     minus = self._eval_preconditioner(x, -pert1)
        #     self._nfev += 4

        # compute the preconditioner point estimate
        diff = loss(x + pert1 + pert2) - plus
        diff -= loss(x - pert1 + pert2) - minus
        diff /= 2 * eps ** 2

        self._nfev += 2

        rank_one = np.outer(delta1, delta2)
        estimate = diff * (rank_one + rank_one.T) / 2

        return estimate

    def _compute_update(self, loss, x, k, eps):
        # compute the perturbations
        if isinstance(self.hessian_resamplings, dict):
            avg = self.hessian_resamplings.get(k, 1)
        else:
            avg = self.hessian_resamplings

        gradient = np.zeros(x.size)
        preconditioner = np.zeros((x.size, x.size))

        # accumulate the number of samples
        for _ in range(avg):
            delta1 = bernoulli_perturbation(x.size, self.perturbation_dims)

            # compute the gradient
            gradient_sample, plus, minus = self._compute_gradient(loss, x, eps, delta1)
            gradient += gradient_sample

            # compute the preconditioner
            if self.second_order:
                delta2 = bernoulli_perturbation(x.size, self.perturbation_dims)
                point_sample = self._point_estimate(loss, x, eps, delta1, delta2, plus, minus)
                preconditioner += point_sample

        # take the mean
        gradient /= avg

        # update the exponentially smoothed average
        if self.second_order:
            preconditioner /= avg
            smoothed = k / (k + 1) * self._moving_avg + 1 / (k + 1) * preconditioner
            self._moving_avg = smoothed

            if k > self.hessian_delay:
                # make the preconditioner SPD
                spd_preconditioner = _make_spd(smoothed, self.regularization)

                # solve for the gradient update
                gradient = np.real(self.lse_solver(spd_preconditioner, gradient))

        return gradient

    def _minimize(self, loss, initial_point):
        # ensure learning rate and perturbation are set
        # this happens only here because for the calibration the loss function is required
        if self.learning_rate is None and self.perturbation is None:
            get_learning_rate, get_perturbation = self.calibrate(loss, initial_point)
            self.learning_rate = get_learning_rate
            self.perturbation = get_perturbation

        if self.learning_rate is None or self.perturbation is None:
            raise ValueError('If one of learning rate or perturbation is set, both must be set.')

        # get iterator
        eta = self.learning_rate()
        eps = self.perturbation()

        # prepare some initials
        x = np.asarray(initial_point)

        if self.initial_hessian is None:
            self._moving_avg = np.identity(x.size)
        else:
            self._moving_avg = self.initial_hessian

        self._nfev = 0

        # if blocking is enabled we need to keep track of the function values
        if self.blocking:
            fx = loss(x)
            self._nfev += 1

        logger.info('=' * 30)
        logger.info('Starting SPSA optimization')
        start = time()

        for k in range(1, self.maxiter + 1):
            iteration_start = time()
            # compute update
            update = self._compute_update(loss, x, k, next(eps))

            # trust region
            if self.trust_region:
                norm = np.linalg.norm(update)
                if norm > 1:  # stop from dividing by 0
                    update = update / norm

            # compute next parameter value
            update = update * next(eta)
            x_next = x - update

            if self.callback is not None:
                callback_args = [x_next,  # next parameters
                                 loss(x_next),  # loss at next parameters
                                 np.linalg.norm(update)]  # size of the update step

            # blocking
            if self.blocking:
                fx_next = loss(x_next)
                self._nfev += 1
                if fx <= fx_next:  # discard update if it didn't improve the loss
                    if self.callback is not None:
                        callback_args += [self._nfev,  # number of function evals
                                          False]  # not accepted
                        self.callback(*callback_args)

                    logger.info('Iteration %s/%s rejected in %s.',
                                k, self.maxiter + 1, time() - iteration_start)
                    continue
                fx = fx_next

            logger.info('Iteration %s/%s done in %s.',
                        k, self.maxiter + 1, time() - iteration_start)

            if self.callback is not None:
                callback_args += [self._nfev,  # number of function evals
                                  True]  # accepted
                self.callback(*callback_args)

            # update parameters
            x = x_next

            # check termination
            if np.linalg.norm(update) < self.tolerance:
                break

        logger.info('SPSA finished in %s', time() - start)
        logger.info('=' * 30)

        return x, loss(x), self._nfev

    def get_support_level(self):
        """Get the support level dictionary."""
        return {
            'gradient': OptimizerSupportLevel.ignored,  # could be supported though
            'bounds': OptimizerSupportLevel.ignored,
            'initial_point': OptimizerSupportLevel.required
        }

    def optimize(self, num_vars, objective_function, gradient_function=None,
                 variable_bounds=None, initial_point=None):
        return self._minimize(objective_function, initial_point)


def bernoulli_perturbation(dim, perturbation_dims=None):
    """Get a Bernoulli random perturbation."""
    if perturbation_dims is None:
        return np.array([1 - 2 * np.random.binomial(1, 0.5) for _ in range(dim)])

    pert = np.array([1 - 2 * np.random.binomial(1, 0.5)
                     for _ in range(perturbation_dims)])
    indices = np.random.choice(list(range(dim)), size=perturbation_dims, replace=False)
    result = np.zeros(dim)
    result[indices] = pert

    return result


def powerseries(eta=0.01, power=2, offset=0):
    """Yield a series decreasing by a powerlaw."""

    n = 1
    while True:
        yield eta / ((n + offset) ** power)
        n += 1


def constant(eta=0.01):
    """Yield a constant series."""

    while True:
        yield eta


def _make_spd(matrix, bias=0.01):
    identity = np.identity(matrix.shape[0])
    psd = scipy.linalg.sqrtm(matrix.dot(matrix))
    return (1 - bias) * psd + bias * identity
