"""Internal NumPy/SciPy port of the Benisse R numerical kernels.

This module is intentionally not a public API. Phase 4d develops and verifies
the mathematical core independently of the Phase 4c bridge and AIRR adapter.
"""

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class HyperParameters:
    lambda1: float
    lambda2: float
    gamma: float
    rho: float
    m: int


@dataclass(frozen=True)
class UpdateAResult:
    matrix: np.ndarray
    success: bool
    status: int
    iterations: int
    objective: float
    message: str
    evaluations: int
    gradient_evaluations: int
    projected_gradient_norm: float
    max_iterations: int


@dataclass(frozen=True)
class ADMMResult:
    q_matrix: np.ndarray
    r_matrix: np.ndarray
    a_matrix: np.ndarray
    sparse_graph: np.ndarray
    iterations: int
    converged: bool
    graph_change_mean: float | None
    graph_change_sd: float | None
    optimizer_results: tuple[UpdateAResult, ...]


class OptimizationError(RuntimeError):
    """Raised when an inner bounded optimization does not converge."""


def _square_matrix(value, name):
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a square matrix")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values")
    return matrix


def _require_same_shape(**matrices):
    shapes = {name: matrix.shape for name, matrix in matrices.items()}
    if len(set(shapes.values())) != 1:
        details = ", ".join(f"{name}={shape}" for name, shape in shapes.items())
        raise ValueError(f"matrices must have the same shape ({details})")


def _require_symmetric(matrix, name, tolerance=1e-12):
    if not np.allclose(matrix, matrix.T, rtol=0, atol=tolerance):
        raise ValueError(f"{name} must be symmetric")


def _validate_hyperparameters(hyperparameters):
    values = {
        "lambda1": hyperparameters.lambda1,
        "lambda2": hyperparameters.lambda2,
        "gamma": hyperparameters.gamma,
        "rho": hyperparameters.rho,
        "m": hyperparameters.m,
    }
    if not all(np.isscalar(value) and np.isfinite(value) for value in values.values()):
        raise ValueError("hyperparameters must be finite scalars")
    if hyperparameters.lambda1 < 0 or hyperparameters.lambda2 < 0:
        raise ValueError("lambda1 and lambda2 must be non-negative")
    if hyperparameters.gamma <= 0 or hyperparameters.rho <= 0:
        raise ValueError("gamma and rho must be positive")
    if not isinstance(hyperparameters.m, Integral) or hyperparameters.m <= 0:
        raise ValueError("m must be a positive integer")


def graph_laplacian(weights):
    weights = _square_matrix(weights, "weights")
    return np.diag(weights.sum(axis=1)) - weights


def update_q(identity, a_matrix, r_matrix, ls_matrix, hyperparameters):
    _validate_hyperparameters(hyperparameters)
    identity = _square_matrix(identity, "identity")
    a_matrix = _square_matrix(a_matrix, "a_matrix")
    r_matrix = _square_matrix(r_matrix, "r_matrix")
    ls_matrix = _square_matrix(ls_matrix, "ls_matrix")
    _require_same_shape(
        identity=identity,
        a_matrix=a_matrix,
        r_matrix=r_matrix,
        ls_matrix=ls_matrix,
    )
    la_matrix = graph_laplacian(a_matrix)
    c_matrix = (
        identity
        + 4 * hyperparameters.gamma * (hyperparameters.lambda2 * ls_matrix + la_matrix)
        + r_matrix / hyperparameters.rho
    )
    eigenvalues, eigenvectors = np.linalg.eigh((c_matrix + c_matrix.T) / 2)
    q_eigenvalues = eigenvalues / 2 + np.sqrt(
        eigenvalues**2 / 4
        + hyperparameters.m / (2 * hyperparameters.rho)
    )
    q_matrix = (eigenvectors * q_eigenvalues) @ eigenvectors.T
    q_matrix = (q_matrix + q_matrix.T) / 2
    return q_matrix, la_matrix


def update_r(identity, la_matrix, r_matrix, q_matrix, ls_matrix, hyperparameters):
    _validate_hyperparameters(hyperparameters)
    identity = _square_matrix(identity, "identity")
    la_matrix = _square_matrix(la_matrix, "la_matrix")
    r_matrix = _square_matrix(r_matrix, "r_matrix")
    q_matrix = _square_matrix(q_matrix, "q_matrix")
    ls_matrix = _square_matrix(ls_matrix, "ls_matrix")
    _require_same_shape(
        identity=identity,
        la_matrix=la_matrix,
        r_matrix=r_matrix,
        q_matrix=q_matrix,
        ls_matrix=ls_matrix,
    )
    return r_matrix - hyperparameters.rho * (
        q_matrix
        - identity
        - 4
        * hyperparameters.gamma
        * (hyperparameters.lambda2 * ls_matrix + la_matrix)
    )


def _active_coordinates(upper_bounds):
    active_upper_triangle = np.triu(upper_bounds != 0, k=1)
    flat_indices = np.flatnonzero(active_upper_triangle.ravel(order="F"))
    return np.unravel_index(flat_indices, upper_bounds.shape, order="F")


def _symmetric_candidate(parameters, active_coordinates, shape):
    candidate = np.zeros(shape, dtype=np.float64)
    rows, columns = active_coordinates
    candidate[rows, columns] = parameters
    candidate[columns, rows] = parameters
    return candidate


def _a_terms(
    parameters,
    active_coordinates,
    shape,
    phi,
    identity,
    r_matrix,
    q_matrix,
    ls_matrix,
    hyperparameters,
):
    candidate = _symmetric_candidate(parameters, active_coordinates, shape)
    la_matrix = graph_laplacian(candidate)
    residual = (
        la_matrix
        - (q_matrix - identity - r_matrix / hyperparameters.rho)
        / (4 * hyperparameters.gamma)
        + hyperparameters.lambda2 * ls_matrix
    )
    objective = (
        0.5
        * hyperparameters.rho
        * np.linalg.norm(
            q_matrix
            - identity
            - 4
            * hyperparameters.gamma
            * (hyperparameters.lambda2 * ls_matrix + la_matrix)
            - r_matrix / hyperparameters.rho,
            ord="fro",
        )
        ** 2
        + np.sum(candidate * phi)
    )
    rows, columns = active_coordinates
    diagonal = np.diag(residual)
    gradient = 2 * phi[rows, columns] + (
        16
        * hyperparameters.rho
        * hyperparameters.gamma**2
        * (
            diagonal[rows]
            + diagonal[columns]
            - residual[rows, columns]
            - residual[columns, rows]
        )
    )
    return objective, gradient


def _optimizer_options(problem_size):
    return {
        "maxiter": 50 if problem_size > 1000 else 100,
        "ftol": 1e7 * np.finfo(np.float64).eps,
        "gtol": 1e-8,
        "maxcor": 5,
        "maxls": 20,
    }


def _projected_gradient_norm(parameters, gradient, lower, upper):
    projected = parameters - np.clip(parameters - gradient, lower, upper)
    return float(np.linalg.norm(projected, ord=np.inf)) if projected.size else 0.0


def update_a(
    phi,
    si_matrix,
    identity,
    a_matrix,
    r_matrix,
    q_matrix,
    ls_matrix,
    hyperparameters,
):
    _validate_hyperparameters(hyperparameters)
    phi = _square_matrix(phi, "phi")
    si_matrix = _square_matrix(si_matrix, "si_matrix")
    identity = _square_matrix(identity, "identity")
    a_matrix = _square_matrix(a_matrix, "a_matrix")
    r_matrix = _square_matrix(r_matrix, "r_matrix")
    q_matrix = _square_matrix(q_matrix, "q_matrix")
    ls_matrix = _square_matrix(ls_matrix, "ls_matrix")
    _require_same_shape(
        phi=phi,
        si_matrix=si_matrix,
        identity=identity,
        a_matrix=a_matrix,
        r_matrix=r_matrix,
        q_matrix=q_matrix,
        ls_matrix=ls_matrix,
    )
    for matrix, name in (
        (phi, "phi"),
        (si_matrix, "si_matrix"),
        (identity, "identity"),
        (a_matrix, "a_matrix"),
        (r_matrix, "r_matrix"),
        (q_matrix, "q_matrix"),
        (ls_matrix, "ls_matrix"),
    ):
        _require_symmetric(matrix, name)

    upper_bounds = hyperparameters.lambda1 * (1 - identity) * si_matrix
    active_coordinates = _active_coordinates(upper_bounds)
    if len(active_coordinates[0]) == 0:
        objective, _ = _a_terms(
            np.empty(0),
            active_coordinates,
            a_matrix.shape,
            phi,
            identity,
            r_matrix,
            q_matrix,
            ls_matrix,
            hyperparameters,
        )
        return UpdateAResult(
            matrix=np.zeros_like(a_matrix),
            success=True,
            status=0,
            iterations=0,
            objective=float(objective),
            message="No active crude-graph edges",
            evaluations=0,
            gradient_evaluations=0,
            projected_gradient_norm=0.0,
            max_iterations=_optimizer_options(a_matrix.shape[0])["maxiter"],
        )

    initial = a_matrix[active_coordinates]
    bounds = [(0.0, bound) for bound in upper_bounds[active_coordinates]]

    # scipy accepts ``(objective, gradient)`` when jac=True, avoiding duplicate
    # construction of the dense Laplacian and residual for each evaluation.
    def objective_and_gradient(parameters):
        return _a_terms(
            parameters,
            active_coordinates,
            a_matrix.shape,
            phi,
            identity,
            r_matrix,
            q_matrix,
            ls_matrix,
            hyperparameters,
        )

    options = _optimizer_options(a_matrix.shape[0])
    optimization = minimize(
        objective_and_gradient,
        initial,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options=options,
    )
    updated = _symmetric_candidate(optimization.x, active_coordinates, a_matrix.shape)
    _, final_gradient = objective_and_gradient(optimization.x)
    lower = np.zeros_like(optimization.x)
    upper = upper_bounds[active_coordinates]
    return UpdateAResult(
        matrix=updated,
        success=bool(optimization.success),
        status=int(optimization.status),
        iterations=int(optimization.nit),
        objective=float(optimization.fun),
        message=str(optimization.message),
        evaluations=int(optimization.nfev),
        gradient_evaluations=int(optimization.njev),
        projected_gradient_norm=_projected_gradient_norm(
            optimization.x, final_gradient, lower, upper
        ),
        max_iterations=options["maxiter"],
    )


def latent_distances(q_matrix, m, gamma):
    q_matrix = _square_matrix(q_matrix, "q_matrix")
    inverse = np.linalg.inv(q_matrix)
    diagonal = np.diag(inverse)
    distances = m * gamma * (
        diagonal[:, None] + diagonal[None, :] - 2 * inverse
    )
    return (distances + distances.T) / 2


def graph_change_mse(current_graph, previous_graph):
    current_graph = _square_matrix(current_graph, "current_graph")
    previous_graph = _square_matrix(previous_graph, "previous_graph")
    if current_graph.shape != previous_graph.shape:
        raise ValueError("current_graph and previous_graph must have the same shape")
    return float(np.sum((current_graph - previous_graph) ** 2) / current_graph.shape[0] ** 2)


def run_admm(
    phi,
    si_matrix,
    ls_matrix,
    hyperparameters,
    *,
    max_iterations,
    stop_cutoff,
):
    """Run the numerical loop in ``R/util.R::Benisse`` from initialized matrices.

    Input preparation remains separate so this kernel can be validated independently
    of the Phase 4c bridge and the eventual AnnData-facing data model.
    """
    _validate_hyperparameters(hyperparameters)
    phi = _square_matrix(phi, "phi")
    si_matrix = _square_matrix(si_matrix, "si_matrix")
    ls_matrix = _square_matrix(ls_matrix, "ls_matrix")
    _require_same_shape(phi=phi, si_matrix=si_matrix, ls_matrix=ls_matrix)
    for matrix, name in (
        (phi, "phi"),
        (si_matrix, "si_matrix"),
        (ls_matrix, "ls_matrix"),
    ):
        _require_symmetric(matrix, name)
    if not isinstance(max_iterations, Integral) or max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")
    if not np.isscalar(stop_cutoff) or not np.isfinite(stop_cutoff) or stop_cutoff < 0:
        raise ValueError("stop_cutoff must be non-negative")

    identity = np.eye(si_matrix.shape[0], dtype=np.float64)
    a_matrix = hyperparameters.lambda1 * (1 - identity) * si_matrix
    q_matrix = si_matrix.copy()
    r_matrix = si_matrix.copy()
    rho = hyperparameters.rho
    graph_history = []
    change_mean = None
    change_sd = None
    converged = False
    optimizer_results = []

    for iteration in range(1, max_iterations + 1):
        iteration_hyper = HyperParameters(
            lambda1=hyperparameters.lambda1,
            lambda2=hyperparameters.lambda2,
            gamma=hyperparameters.gamma,
            rho=rho,
            m=hyperparameters.m,
        )
        q_matrix, la_matrix = update_q(
            identity, a_matrix, r_matrix, ls_matrix, iteration_hyper
        )
        r_matrix = update_r(
            identity,
            la_matrix,
            r_matrix,
            q_matrix,
            ls_matrix,
            iteration_hyper,
        )
        optimizer_result = update_a(
            phi,
            si_matrix,
            identity,
            a_matrix,
            r_matrix,
            q_matrix,
            ls_matrix,
            iteration_hyper,
        )
        if not optimizer_result.success:
            raise OptimizationError(
                f"A optimization failed at ADMM iteration {iteration}: "
                f"status={optimizer_result.status}, message={optimizer_result.message}"
            )
        optimizer_results.append(optimizer_result)
        a_matrix = optimizer_result.matrix
        rho *= 2 / (1 + np.sqrt(5))

        sparse_graph = a_matrix > 0
        graph_history.append(sparse_graph)
        if len(graph_history) > 11:
            graph_history.pop(0)
        if iteration > 10:
            changes = np.asarray(
                [
                    graph_change_mse(current, previous)
                    for previous, current in zip(graph_history, graph_history[1:])
                ]
            )
            change_mean = float(changes.mean())
            # R's sd() uses the sample standard deviation.
            change_sd = float(changes.std(ddof=1))
            if change_mean < stop_cutoff and change_sd < 1e-4:
                converged = True
                break

    return ADMMResult(
        q_matrix=q_matrix,
        r_matrix=r_matrix,
        a_matrix=a_matrix,
        sparse_graph=sparse_graph,
        iterations=iteration,
        converged=converged,
        graph_change_mean=change_mean,
        graph_change_sd=change_sd,
        optimizer_results=tuple(optimizer_results),
    )
