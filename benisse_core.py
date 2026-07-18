"""Internal NumPy/SciPy port of the Benisse R numerical kernels.

This module is intentionally not a public API. Phase 4d develops and verifies
the mathematical core independently of the Phase 4c bridge and AIRR adapter.
"""

from dataclasses import dataclass

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


def graph_laplacian(weights):
    weights = _square_matrix(weights, "weights")
    return np.diag(weights.sum(axis=1)) - weights


def update_q(identity, a_matrix, r_matrix, ls_matrix, hyperparameters):
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
    flat_indices = np.flatnonzero(upper_bounds.ravel(order="F") != 0)
    return np.unravel_index(flat_indices, upper_bounds.shape, order="F")


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
    candidate = np.zeros(shape, dtype=np.float64)
    candidate[active_coordinates] = parameters
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
    # R recycles the vector returned by ``diag(U)`` down every matrix column in
    # ``-U-t(U)+2*diag(U)``. Broadcasting the diagonal as a column reproduces
    # that behavior; constructing a diagonal matrix would change the optimizer.
    gradient_matrix = 2 * phi + 16 * hyperparameters.rho * hyperparameters.gamma**2 * (
        -residual - residual.T + 2 * np.diag(residual)[:, None]
    )
    return objective, gradient_matrix[active_coordinates]


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
        )

    initial = a_matrix[active_coordinates]
    bounds = [(0.0, bound) for bound in upper_bounds[active_coordinates]]

    def objective(parameters):
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
        )[0]

    def gradient(parameters):
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
        )[1]

    max_iterations = 50 if a_matrix.shape[0] > 1000 else 100
    optimization = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=gradient,
        bounds=bounds,
        options={
            "maxiter": max_iterations,
            "ftol": 1e7 * np.finfo(np.float64).eps,
            "gtol": 0.0,
            "maxcor": 5,
            "maxls": 20,
        },
    )
    updated = np.zeros_like(a_matrix)
    updated[active_coordinates] = optimization.x
    updated = (updated + updated.T) / 2
    return UpdateAResult(
        matrix=updated,
        success=bool(optimization.success),
        status=int(optimization.status),
        iterations=int(optimization.nit),
        objective=float(optimization.fun),
        message=str(optimization.message),
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
    phi = _square_matrix(phi, "phi")
    si_matrix = _square_matrix(si_matrix, "si_matrix")
    ls_matrix = _square_matrix(ls_matrix, "ls_matrix")
    _require_same_shape(phi=phi, si_matrix=si_matrix, ls_matrix=ls_matrix)
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")
    if stop_cutoff < 0:
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
        a_matrix = update_a(
            phi,
            si_matrix,
            identity,
            a_matrix,
            r_matrix,
            q_matrix,
            ls_matrix,
            iteration_hyper,
        ).matrix
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
    )
