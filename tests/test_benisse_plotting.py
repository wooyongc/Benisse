"""Scientific and rendering checks for Python Benisse post-analysis."""

import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import airr_adapter as aa  # noqa: E402
import benisse_plotting as bp  # noqa: E402


EXAMPLE = REPO_ROOT / "example"
RSCRIPT = shutil.which("Rscript")
ORACLE_R = REPO_ROOT / "tests" / "fixtures" / "export_post_analysis_oracle.R"


def _small_plot_data(with_embedding=True):
    annotation = pd.DataFrame(
        {
            "v_gene": ["V1", "V1", "V1", "V2"],
            "j_gene": ["J1", "J1", "J1", "J2"],
            "cdr3": ["C1", "C2", "C3", "C4"],
            "clsize": [1, 4, 2, 1],
        }
    )
    network = aa.BenisseNetworkResult(
        node_ids=["V1_C1_J1", "V1_C2_J1", "V1_C3_J1", "V2_C4_J2"],
        row=np.array([0, 1], dtype=np.int64),
        col=np.array([1, 2], dtype=np.int64),
        weight=np.array([0.5, 1.0]),
    )
    coordinates = np.array([[0, 0], [1, 0], [3, 0], [7, 0]], dtype=float)
    latent = np.abs(coordinates[:, 0, None] - coordinates[:, 0])
    embedding = (
        np.array([[0, 1, 0], [1, 1, 0], [2, 0, 1], [4, 0, 0]], dtype=float)
        if with_embedding else None
    )
    return bp.BenissePlotData(network, annotation, latent, embedding)


def test_adjacency_crude_graph_components_and_retention():
    data = _small_plot_data()
    adjacency = bp.symmetric_adjacency(data.network).toarray()
    candidate = bp.crude_graph(data.annotation)
    labels, sizes = bp.component_labels(data.network)

    assert np.array_equal(adjacency, adjacency.T)
    assert adjacency[0, 1] == 0.5 and adjacency[1, 2] == 1.0
    assert np.count_nonzero(np.triu(candidate, 1)) == 3
    assert labels.tolist() == [0, 0, 0, 1]
    assert sizes.tolist() == [3, 1]
    assert bp.edge_retention_ratio(data) == pytest.approx(2 / 3)


def test_latent_distance_groups_use_unique_pairs():
    groups = bp.latent_distance_groups(_small_plot_data())
    assert groups[bp.DISTANCE_LABELS[0]].tolist() == [1.0, 2.0]
    assert groups[bp.DISTANCE_LABELS[1]].tolist() == [3.0]
    assert sorted(groups[bp.DISTANCE_LABELS[2]].tolist()) == [4.0, 6.0, 7.0]


def test_pca_coordinates_are_deterministic_and_handle_one_node():
    embedding = _small_plot_data().embedding
    assert np.array_equal(bp.pca_coordinates(embedding), bp.pca_coordinates(embedding))
    assert bp.pca_coordinates(np.array([[2.0, 3.0]])).shape == (1, 2)
    with pytest.raises(ValueError, match="non-empty"):
        bp.pca_coordinates(np.empty((0, 2)))
    with pytest.raises(ValueError, match="method must"):
        bp.embedding_coordinates(embedding, method="unknown")


def test_clone_expression_distances_are_symmetric_and_node_ordered():
    expression = pd.DataFrame(
        np.array(
            [
                [0, 1, 2, 3, 4, 5],
                [5, 4, 3, 2, 1, 0],
                [0, 0, 1, 1, 3, 3],
                [1, 4, 1, 4, 1, 4],
                [2, 3, 5, 7, 11, 13],
                [1, 1, 2, 3, 5, 8],
                [8, 5, 3, 2, 1, 1],
                [1, 0, 0, 1, 0, 1],
                [2, 4, 8, 16, 32, 64],
                [3, 1, 4, 1, 5, 9],
            ],
            dtype=float,
        ),
        columns=[f"cell{i}" for i in range(6)],
    )
    labels = np.array(["clone_b", "clone_a", "clone_b", "clone_c", "clone_a", "clone_c"])
    distances = bp.clone_expression_distances(
        expression, labels, ["clone_c", "clone_b", "clone_a"]
    )
    assert distances.shape == (3, 3)
    assert np.allclose(distances, distances.T)
    assert np.all(np.diag(distances) >= 0)
    assert np.isfinite(distances).all()
    with pytest.raises(ValueError, match="missing 1"):
        bp.clone_expression_distances(expression, labels, ["missing"])


def test_coupling_data_matches_direct_spearman_and_handles_empty_graph():
    data = _small_plot_data()
    expression = data.latent_distances * 2
    embedding = data.latent_distances * 3
    reduced, correlations = bp.coupling_data(
        expression, data.latent_distances, embedding, data.network
    )
    assert len(reduced) == 4  # two undirected edges, retained as R-style directed entries
    assert correlations == {"expression_latent": 1.0, "expression_embedding": 1.0}

    empty = aa.BenisseNetworkResult(
        node_ids=data.network.node_ids,
        row=np.array([], dtype=np.int64), col=np.array([], dtype=np.int64),
        weight=np.array([], dtype=float),
    )
    reduced, correlations = bp.coupling_data(
        expression, data.latent_distances, embedding, empty
    )
    assert reduced.empty
    assert np.isnan(correlations["expression_latent"])


@pytest.mark.parametrize("color_by", ["clone_size", "vj_family", "graph_component"])
def test_connection_plot_color_modes_render(color_by):
    axis = bp.plot_embedding_network(_small_plot_data(), color_by=color_by)
    assert axis.get_title() == "Benisse clonotype network"
    assert len(axis.collections) >= 2
    plt.close(axis.figure)


def test_all_plot_types_write_nonempty_pdf(tmp_path):
    data = _small_plot_data()
    axes = [
        bp.plot_embedding_network(data),
        bp.plot_latent_distance_groups(data),
        bp.plot_network_summary(data)[0],
    ]
    reduced, correlations = bp.coupling_data(
        data.latent_distances * 2,
        data.latent_distances,
        data.latent_distances * 3,
        data.network,
    )
    axes.append(bp.plot_coupling_correlation(reduced, correlations))
    for index, axis in enumerate(axes):
        path = tmp_path / f"plot_{index}.pdf"
        axis.figure.savefig(path)
        assert path.read_bytes().startswith(b"%PDF")
        assert path.stat().st_size > 1_000
        plt.close(axis.figure)


def test_read_committed_example_plot_data():
    data = bp.read_plot_data(EXAMPLE, encoded_csv=EXAMPLE / "encoded_10x_NSCLC.csv")
    assert data.network.n_nodes == 1494
    assert data.network.n_edges == 1691
    assert data.annotation.shape[0] == 1494
    assert data.embedding.shape == (1494, 20)
    assert data.latent_distances.shape == (1494, 1494)
    assert np.isfinite(bp.edge_retention_ratio(data))


@pytest.mark.skipif(RSCRIPT is None, reason="Rscript is required for R post-analysis parity")
def test_expression_distance_and_correlation_match_committed_r_oracle(tmp_path):
    subprocess.run(
        [
            RSCRIPT,
            str(ORACLE_R),
            str(EXAMPLE / "Benisse_results.RData"),
            str(EXAMPLE / "encoded_10x_NSCLC.csv"),
            str(tmp_path),
        ],
        cwd=REPO_ROOT,
        check=True,
        timeout=120,
    )
    data = bp.read_plot_data(EXAMPLE, encoded_csv=EXAMPLE / "encoded_10x_NSCLC.csv")
    expression = bp.clone_expression_distances(
        EXAMPLE / "cleaned_exp.txt",
        EXAMPLE / "clonality_label.txt",
        data.network.node_ids,
    )
    r_expression = np.loadtxt(tmp_path / "master_dist_e.txt")
    r_candidate = np.loadtxt(tmp_path / "si.txt").astype(bool)
    assert np.allclose(expression, r_expression, rtol=2e-6, atol=2e-8)
    assert np.array_equal(bp.crude_graph(data.annotation), r_candidate)

    _, correlations = bp.coupling_data(
        expression,
        data.latent_distances,
        bp.bcr_embedding_distances(data.embedding),
        data.network,
    )
    r_correlations = np.loadtxt(tmp_path / "correlations.txt")
    assert correlations["expression_latent"] == pytest.approx(r_correlations[0], abs=2e-8)
    assert correlations["expression_embedding"] == pytest.approx(r_correlations[1], abs=2e-8)
