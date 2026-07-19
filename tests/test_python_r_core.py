"""Small-fixture tests for the experimental corrected Phase 4d core."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import benisse_core as core  # noqa: E402
from benisse_core import (  # noqa: E402
    HyperParameters,
    OptimizationError,
    UpdateAResult,
    graph_change_mse,
    graph_laplacian,
    latent_distances,
    run_admm,
    update_a,
    update_q,
    update_r,
)


FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "r_core_golden.json"


@pytest.fixture(scope="module")
def r_golden():
    return json.loads(FIXTURE_PATH.read_text())


@pytest.fixture(scope="module")
def small_case(r_golden):
    inputs = {
        name: np.asarray(value, dtype=np.float64)
        for name, value in r_golden["small"]["inputs"].items()
    }
    expected = {
        name: np.asarray(value, dtype=np.float64)
        if isinstance(value, list)
        else value
        for name, value in r_golden["small"]["expected"].items()
    }
    hyper = HyperParameters(**r_golden["hyperparameters"])
    return inputs, expected, hyper


def _updated_q_and_r(inputs, hyper):
    q_matrix, la_matrix = update_q(
        inputs["I"], inputs["A"], inputs["R"], inputs["LS"], hyper
    )
    r_matrix = update_r(
        inputs["I"], la_matrix, inputs["R"], q_matrix, inputs["LS"], hyper
    )
    return q_matrix, la_matrix, r_matrix


def _permute(matrix, order):
    return matrix[np.ix_(order, order)]


def test_linear_algebra_updates_match_corrected_r_fixture(small_case):
    inputs, expected, hyper = small_case
    q_matrix, la_matrix, r_matrix = _updated_q_and_r(inputs, hyper)

    np.testing.assert_allclose(
        graph_laplacian(inputs["A"]), expected["LA"], rtol=0, atol=1e-14
    )
    np.testing.assert_allclose(la_matrix, expected["LA"], rtol=0, atol=1e-14)
    np.testing.assert_allclose(q_matrix, expected["Q"], rtol=0, atol=1e-13)
    np.testing.assert_allclose(r_matrix, expected["R"], rtol=0, atol=1e-13)


def test_symmetric_a_gradient_matches_central_finite_difference(small_case):
    inputs, _, hyper = small_case
    q_matrix, _, r_matrix = _updated_q_and_r(inputs, hyper)
    upper = hyper.lambda1 * (1 - inputs["I"]) * inputs["SI"]
    coordinates = core._active_coordinates(upper)
    assert len(coordinates[0]) == np.count_nonzero(inputs["SI"]) // 2
    assert np.all(coordinates[0] < coordinates[1])
    parameters = np.linspace(0.12, 0.36, len(coordinates[0]))

    def terms(values):
        return core._a_terms(
            values,
            coordinates,
            inputs["A"].shape,
            inputs["phi"],
            inputs["I"],
            r_matrix,
            q_matrix,
            inputs["LS"],
            hyper,
        )

    objective, analytic = terms(parameters)
    assert np.isfinite(objective)
    epsilon = 1e-6
    numeric = np.empty_like(parameters)
    for index in range(len(parameters)):
        step = np.zeros_like(parameters)
        step[index] = epsilon
        numeric[index] = (terms(parameters + step)[0] - terms(parameters - step)[0]) / (
            2 * epsilon
        )
    np.testing.assert_allclose(analytic, numeric, rtol=1e-7, atol=1e-6)


def test_corrected_update_a_matches_r_and_optimizer_succeeds(r_golden, small_case):
    inputs, expected, hyper = small_case
    r_optimizer = r_golden["small"]["optimizer"]
    q_matrix, _, r_matrix = _updated_q_and_r(inputs, hyper)
    result = update_a(
        inputs["phi"],
        inputs["SI"],
        inputs["I"],
        inputs["A"],
        r_matrix,
        q_matrix,
        inputs["LS"],
        hyper,
    )

    assert result.success
    assert result.status == 0
    assert result.projected_gradient_norm < 1e-4
    assert r_optimizer["convergence"] == 0
    assert r_optimizer["projected_gradient_norm"] < 1e-4
    assert result.max_iterations == r_golden["optimizer_policy"]["small_maxit"]
    np.testing.assert_allclose(result.matrix, expected["A"], rtol=0, atol=1e-12)
    assert np.array_equal(result.matrix > 0, expected["A"] > 0)
    assert np.array_equal(result.matrix, result.matrix.T)
    assert np.all(np.diag(result.matrix) == 0)
    assert np.all(result.matrix >= 0)
    assert np.all(result.matrix <= hyper.lambda1)
    assert np.all(result.matrix[inputs["SI"] == 0] == 0)


def test_one_admm_iteration_matches_corrected_r(r_golden, small_case):
    inputs, _, hyper = small_case
    expected = r_golden["small"]["admm_one_iteration"]
    result = run_admm(
        inputs["phi"], inputs["SI"], inputs["LS"], hyper,
        max_iterations=1, stop_cutoff=0,
    )

    assert result.iterations == 1
    assert not result.converged
    assert len(result.optimizer_results) == 1
    assert result.optimizer_results[0].success
    assert expected["optimizer_convergence"] == 0
    np.testing.assert_allclose(result.q_matrix, expected["Q"], rtol=0, atol=1e-13)
    np.testing.assert_allclose(result.r_matrix, expected["R"], rtol=0, atol=1e-13)
    np.testing.assert_allclose(result.a_matrix, expected["A"], rtol=0, atol=1e-12)


def test_small_admm_matches_corrected_r_and_scientific_invariants(r_golden, small_case):
    inputs, _, hyper = small_case
    expected = r_golden["small"]["admm"]
    result = run_admm(
        inputs["phi"], inputs["SI"], inputs["LS"], hyper,
        max_iterations=30, stop_cutoff=1e-10,
    )
    latent = latent_distances(result.q_matrix, hyper.m, hyper.gamma)

    assert result.converged
    assert result.iterations == expected["iterations"]
    assert set(expected["optimizer_convergence"]) == {0}
    assert all(optimizer.success for optimizer in result.optimizer_results)
    assert np.array_equal(result.sparse_graph, np.asarray(expected["sparse_graph"]))
    np.testing.assert_allclose(result.a_matrix, expected["A"], rtol=0, atol=1e-11)
    np.testing.assert_allclose(result.q_matrix, expected["Q"], rtol=0, atol=3e-11)
    np.testing.assert_allclose(result.r_matrix, expected["R"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(latent, expected["latent"], rtol=0, atol=3e-11)

    assert np.array_equal(result.a_matrix, result.a_matrix.T)
    assert np.all(result.a_matrix[inputs["SI"] == 0] == 0)
    assert np.linalg.eigvalsh(result.q_matrix).min() > 0
    assert np.all(np.isfinite(latent))
    assert np.allclose(latent, latent.T, rtol=0, atol=1e-13)
    assert np.allclose(np.diag(latent), 0, rtol=0, atol=1e-13)
    assert latent.min() >= -1e-13


def test_legacy_r_and_corrected_v2_topology_gap_is_intentional(r_golden):
    """Freeze the v1 bug-for-bug graph separately from corrected v2 behavior."""
    corrected = np.asarray(r_golden["small"]["admm"]["sparse_graph"], dtype=bool)
    legacy = np.asarray(
        r_golden["small"]["legacy_production_gap"]["sparse_graph"], dtype=bool
    )
    corrected_edges = {tuple(item) for item in np.argwhere(np.triu(corrected, k=1))}
    legacy_edges = {tuple(item) for item in np.argwhere(np.triu(legacy, k=1))}
    assert corrected_edges == {(0, 1), (2, 3)}
    assert legacy_edges == {(0, 1), (0, 2), (2, 3)}
    assert corrected_edges < legacy_edges
    assert r_golden["small"]["admm"]["iterations"] == 23
    assert r_golden["small"]["legacy_production_gap"]["iterations"] == 18


def test_solution_is_equivariant_to_node_permutation(small_case):
    inputs, _, hyper = small_case
    baseline = run_admm(
        inputs["phi"], inputs["SI"], inputs["LS"], hyper,
        max_iterations=30, stop_cutoff=1e-10,
    )
    order = np.array([2, 0, 3, 1])
    inverse = np.argsort(order)
    permuted = run_admm(
        _permute(inputs["phi"], order),
        _permute(inputs["SI"], order),
        _permute(inputs["LS"], order),
        hyper,
        max_iterations=30,
        stop_cutoff=1e-10,
    )

    np.testing.assert_allclose(
        _permute(permuted.a_matrix, inverse), baseline.a_matrix, rtol=0, atol=1e-10
    )
    np.testing.assert_allclose(
        _permute(permuted.q_matrix, inverse), baseline.q_matrix, rtol=0, atol=1e-10
    )
    assert np.array_equal(
        _permute(permuted.sparse_graph, inverse), baseline.sparse_graph
    )


def test_alternate_small_sparse_case_remains_valid():
    n = 8
    indices = np.arange(n)
    phi = (indices[:, None] - indices[None, :]) ** 2
    crude = np.zeros((n, n))
    for index in range(n):
        crude[index, (index + 1) % n] = 1
        crude[(index + 1) % n, index] = 1
    overlap = crude / (1 + phi)
    ls_matrix = -(np.diag(overlap.sum(axis=1)) - overlap) / crude.sum()
    hyper = HyperParameters(lambda1=0.5, lambda2=0.7, gamma=0.4, rho=1.0, m=2)

    result = run_admm(
        phi, crude, ls_matrix, hyper, max_iterations=5, stop_cutoff=0
    )
    assert len(result.optimizer_results) == 5
    assert all(optimizer.success for optimizer in result.optimizer_results)
    assert np.array_equal(result.a_matrix, result.a_matrix.T)
    assert np.all(result.a_matrix[crude == 0] == 0)
    assert np.linalg.eigvalsh(result.q_matrix).min() > 0


def test_empty_crude_graph_converges_without_optimizer_work():
    zero = np.zeros((3, 3))
    hyper = HyperParameters(lambda1=1, lambda2=1, gamma=1, rho=1, m=2)
    result = run_admm(zero, zero, zero, hyper, max_iterations=12, stop_cutoff=1e-10)

    assert result.converged
    assert result.iterations == 11
    assert all(optimizer.success for optimizer in result.optimizer_results)
    assert all(optimizer.evaluations == 0 for optimizer in result.optimizer_results)
    assert not result.sparse_graph.any()


def test_run_admm_rejects_failed_inner_solve(monkeypatch):
    failed = UpdateAResult(
        matrix=np.zeros((2, 2)), success=False, status=2, iterations=0,
        objective=0.0, message="ABNORMAL", evaluations=1, gradient_evaluations=1,
        projected_gradient_norm=1.0, max_iterations=100,
    )
    monkeypatch.setattr(core, "update_a", lambda *args, **kwargs: failed)
    zero = np.zeros((2, 2))
    hyper = HyperParameters(lambda1=1, lambda2=1, gamma=1, rho=1, m=1)
    with pytest.raises(OptimizationError, match="ADMM iteration 1.*status=2"):
        run_admm(zero, zero, zero, hyper, max_iterations=1, stop_cutoff=0)


def test_optimizer_policy_selects_large_branch_without_large_allocation(r_golden):
    policy = r_golden["optimizer_policy"]
    assert core._optimizer_options(1000)["maxiter"] == policy["small_maxit"]
    assert core._optimizer_options(1001)["maxiter"] == policy["large_maxit"]


@pytest.mark.parametrize(
    "hyper",
    [
        HyperParameters(-1, 1, 1, 1, 1),
        HyperParameters(1, -1, 1, 1, 1),
        HyperParameters(1, 1, 0, 1, 1),
        HyperParameters(1, 1, 1, 0, 1),
        HyperParameters(1, 1, 1, 1, 1.5),
        HyperParameters(1, 1, np.nan, 1, 1),
    ],
)
def test_invalid_hyperparameters_are_rejected(hyper):
    zero = np.zeros((2, 2))
    with pytest.raises(ValueError):
        run_admm(zero, zero, zero, hyper, max_iterations=1, stop_cutoff=0)


def test_invalid_controller_and_structural_inputs_are_rejected(small_case):
    inputs, _, hyper = small_case
    asymmetric = inputs["phi"].copy()
    asymmetric[0, 1] += 1
    with pytest.raises(ValueError, match="symmetric"):
        run_admm(asymmetric, inputs["SI"], inputs["LS"], hyper,
                 max_iterations=1, stop_cutoff=0)
    with pytest.raises(ValueError, match="max_iterations"):
        run_admm(inputs["phi"], inputs["SI"], inputs["LS"], hyper,
                 max_iterations=1.5, stop_cutoff=0)
    with pytest.raises(ValueError, match="stop_cutoff"):
        run_admm(inputs["phi"], inputs["SI"], inputs["LS"], hyper,
                 max_iterations=1, stop_cutoff=np.nan)


def test_latent_distance_and_convergence_helpers_match_r(small_case):
    inputs, expected, hyper = small_case
    q_matrix, _, _ = _updated_q_and_r(inputs, hyper)
    np.testing.assert_allclose(
        latent_distances(q_matrix, hyper.m, hyper.gamma),
        expected["latent"], rtol=0, atol=1e-13,
    )
    current = np.array([[0.0, 1.0], [-1.0, 0.0]])
    assert graph_change_mse(current, np.zeros((2, 2))) == expected["graph_change_mse"]


def test_shape_mismatch_is_rejected(small_case):
    inputs, _, hyper = small_case
    with pytest.raises(ValueError, match="same shape"):
        update_q(np.eye(3), inputs["A"], inputs["R"], inputs["LS"], hyper)
