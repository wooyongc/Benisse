"""Component parity tests for the Phase 4d NumPy/SciPy R-core port."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benisse_core import (  # noqa: E402
    HyperParameters,
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


def test_linear_algebra_updates_match_r_golden(small_case):
    inputs, expected, hyper = small_case
    q_matrix, la_matrix = update_q(
        inputs["I"], inputs["A"], inputs["R"], inputs["LS"], hyper
    )
    r_matrix = update_r(
        inputs["I"], la_matrix, inputs["R"], q_matrix, inputs["LS"], hyper
    )

    np.testing.assert_allclose(graph_laplacian(inputs["A"]), expected["LA"], atol=1e-14)
    np.testing.assert_allclose(la_matrix, expected["LA"], atol=1e-14)
    np.testing.assert_allclose(q_matrix, expected["Q"], atol=1e-13)
    np.testing.assert_allclose(r_matrix, expected["R"], atol=1e-13)


def test_bounded_a_update_matches_r_golden(small_case):
    inputs, expected, hyper = small_case
    q_matrix, la_matrix = update_q(
        inputs["I"], inputs["A"], inputs["R"], inputs["LS"], hyper
    )
    r_matrix = update_r(
        inputs["I"], la_matrix, inputs["R"], q_matrix, inputs["LS"], hyper
    )
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

    # R's gradient has column-recycling semantics. Matching it is essential:
    # a conventional diagonal-matrix translation changes these weights by >0.2.
    np.testing.assert_allclose(result.matrix, expected["A"], atol=1e-12)
    assert np.array_equal(result.matrix > 0, expected["A"] > 0)
    assert np.allclose(result.matrix, result.matrix.T)
    assert np.all(result.matrix[inputs["SI"] == 0] == 0)


def test_latent_distance_and_convergence_match_r_golden(small_case):
    inputs, expected, hyper = small_case
    q_matrix, _ = update_q(
        inputs["I"], inputs["A"], inputs["R"], inputs["LS"], hyper
    )
    np.testing.assert_allclose(
        latent_distances(q_matrix, hyper.m, hyper.gamma),
        expected["latent"],
        atol=1e-13,
    )
    current = np.array([[0.0, 1.0], [-1.0, 0.0]])
    previous = np.zeros((2, 2))
    assert graph_change_mse(current, previous) == expected["graph_change_mse"]


def test_admm_controller_one_iteration_has_r_structure_and_bounded_drift(
    r_golden, small_case
):
    inputs, _, hyper = small_case
    expected = r_golden["small"]["admm_one_iteration"]
    result = run_admm(
        inputs["phi"],
        inputs["SI"],
        inputs["LS"],
        hyper,
        max_iterations=1,
        stop_cutoff=0,
    )

    assert result.iterations == 1
    assert not result.converged
    np.testing.assert_allclose(result.q_matrix, expected["Q"], atol=1e-13)
    np.testing.assert_allclose(result.r_matrix, expected["R"], atol=1e-13)
    expected_a = np.asarray(expected["A"])
    assert np.array_equal(result.a_matrix > 0, expected_a > 0)
    assert np.max(np.abs(result.a_matrix - expected_a)) <= 0.03


def test_large_problem_uses_r_large_branch_and_matches_weights(r_golden):
    fixture = r_golden["large_optimizer_branch"]
    n = fixture["n"]
    hyper = HyperParameters(**r_golden["hyperparameters"])
    identity = np.eye(n)
    crude = np.zeros((n, n))
    for row, column in fixture["active_edges"]:
        crude[row - 1, column - 1] = 1.0
    initial = 0.8 * crude
    target = 0.25 * crude
    q_matrix = identity + 4 * hyper.gamma * graph_laplacian(target)
    zero = np.zeros((n, n))

    result = update_a(
        zero, crude, identity, initial, zero, q_matrix, zero, hyper
    )
    rows, columns = zip(
        *((row - 1, column - 1) for row, column in fixture["active_edges"])
    )
    np.testing.assert_allclose(
        result.matrix[rows, columns], fixture["expected_weights"], atol=1e-12
    )
    assert result.iterations <= fixture["expected_maxit"]


def test_core_rejects_shape_mismatch(small_case):
    inputs, _, hyper = small_case
    with pytest.raises(ValueError, match="same shape"):
        update_q(np.eye(3), inputs["A"], inputs["R"], inputs["LS"], hyper)


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("BENISSE_RUN_SLOW_TESTS") != "1" or shutil.which("Rscript") is None,
    reason="set BENISSE_RUN_SLOW_TESTS=1 and install R for the full R-core oracle",
)
def test_complete_example_edge_set_is_near_exact_to_corrected_r_oracle(tmp_path):
    subprocess.run(
        [
            shutil.which("Rscript"),
            "tests/fixtures/export_example_core_inputs.R",
            str(tmp_path),
        ],
        cwd=REPO_ROOT,
        check=True,
        timeout=120,
    )
    n = int((tmp_path / "n.txt").read_text())

    def read_matrix(name, dtype=np.float64):
        return np.fromfile(tmp_path / name, dtype=dtype).reshape((n, n), order="F")

    phi = read_matrix("phi.bin")
    si_matrix = read_matrix("si.bin")
    ls_matrix = read_matrix("ls.bin")
    expected_a = read_matrix("expected_a.bin")
    expected_edges = read_matrix("expected_edges.bin", dtype=np.int32).astype(bool)
    hyper = HyperParameters(lambda1=1, lambda2=1610, gamma=1, rho=1, m=10)

    result = run_admm(
        phi,
        si_matrix,
        ls_matrix,
        hyper,
        max_iterations=100,
        stop_cutoff=1e-10,
    )

    assert result.converged
    assert result.iterations == 33
    intersection = np.count_nonzero(result.sparse_graph & expected_edges)
    union = np.count_nonzero(result.sparse_graph | expected_edges)
    mismatches = np.count_nonzero(result.sparse_graph != expected_edges)
    diagnostics = (
        f"python_edges={np.count_nonzero(result.sparse_graph)}, "
        f"r_edges={np.count_nonzero(expected_edges)}, mismatches={mismatches}, "
        f"jaccard={intersection / union if union else 1.0:.12f}, "
        f"max_abs_weight_error={np.max(np.abs(result.a_matrix - expected_a)):.6g}"
    )
    # R's optim() and SciPy's L-BFGS-B make a different boundary decision for
    # one undirected edge in this 1,494-clone case. Keep this exception narrow:
    # any second edge flip or lower overlap fails the scientific parity gate.
    assert mismatches <= 2, diagnostics
    assert intersection / union >= 0.999, diagnostics

    matching_support = result.sparse_graph == expected_edges
    matching_weight_error = np.abs(
        result.a_matrix[matching_support] - expected_a[matching_support]
    )
    # One additional retained undirected edge differs slightly in weight across
    # the two L-BFGS-B implementations. All other shared-support entries agree
    # to 1e-8; cap both the count and magnitude of this known exception.
    assert np.count_nonzero(matching_weight_error > 1e-8) <= 2, diagnostics
    assert matching_weight_error.max() <= 0.003, diagnostics
