"""Tests for guarded, local-only real-data validation tooling."""

import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import awkward as ak
import numpy as np
import pandas as pd
import pytest
from scipy import sparse


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import airr_adapter as aa  # noqa: E402
import real_data_validation as rdv  # noqa: E402
from benisse_core import ADMMResult, HyperParameters  # noqa: E402


anndata = pytest.importorskip("anndata")
mudata = pytest.importorskip("mudata")
AP4_PATH = REPO_ROOT / "data" / "external" / "stephenson2021_ap4_203.h5mu"
EXAMPLE = REPO_ROOT / "example"


def _chain(sequence_id, junction_aa, v_call, j_call, count):
    return {
        "locus": "IGH",
        "productive": True,
        "junction_aa": junction_aa,
        "junction": "TGT" + "GCC" * 3 + "TGG",
        "sequence_id": sequence_id,
        "v_call": v_call,
        "d_call": "IGHD1",
        "j_call": j_call,
        "c_call": "IGHM",
        "cdr3_aa": junction_aa[1:-1],
        "duplicate_count": count,
        "consensus_count": count * 10,
    }


def _write_small_mudata(path):
    cells = ["cell_A-1", "cell_B-1", "cell_C-1"]
    airr = anndata.AnnData(np.zeros((3, 0), dtype=np.float32))
    airr.obs_names = cells
    airr.obsm["airr"] = ak.Array(
        [
            [_chain("a_contig", "CARAAW", "IGHV1", "IGHJ1", 5)],
            [_chain("b_contig", "CARBBW", "IGHV2", "IGHJ2", 4)],
            [_chain("c_contig", "CARCCW", "IGHV3", "IGHJ3", 3)],
        ]
    )
    gex = anndata.AnnData(sparse.csr_matrix(np.arange(12).reshape(3, 4)))
    gex.obs_names = cells
    gex.var_names = ["g1", "g2", "g3", "g4"]
    gex.obs["sample_id"] = pd.Categorical(["sample_A", "sample_A", "sample_B"])
    mdata = mudata.MuData({"gex": gex, "airr": airr})
    mdata.write_h5mu(path)


def test_prepare_mudata_validation_inputs_is_sample_scoped_and_r_compatible(tmp_path):
    source = tmp_path / "small.h5mu"
    _write_small_mudata(source)
    prepared = rdv.prepare_mudata_validation_inputs(
        source, tmp_path / "prepared", sample_id="sample_A", max_nodes=2
    )

    assert prepared.selected_cells == 2
    assert prepared.estimated_nodes == 2
    expression = pd.read_csv(prepared.expression_csv, index_col=0)
    contigs = pd.read_csv(prepared.contigs_csv)
    encoder = pd.read_csv(prepared.encoder_csv, index_col=0)
    assert expression.shape == (4, 2)
    assert list(expression.columns) == ["cell_A-1", "cell_B-1"]
    assert list(contigs["is_cell"].astype(str)) == ["True", "True"]
    assert list(contigs["chain"]) == ["IGH", "IGH"]
    assert list(encoder.columns) == ["contigs", "cdr3"]
    assert (tmp_path / "prepared" / "input_metadata.json").exists()


def test_prepare_mudata_validation_inputs_enforces_node_guard(tmp_path):
    source = tmp_path / "small.h5mu"
    _write_small_mudata(source)
    with pytest.raises(ValueError, match="refusing 2 clone nodes"):
        rdv.prepare_mudata_validation_inputs(
            source, tmp_path / "prepared", sample_id="sample_A", max_nodes=1
        )


def test_network_conversion_and_comparison_report():
    hyper = HyperParameters(lambda1=1, lambda2=3, gamma=1, rho=1, m=2)
    corrected_admm = SimpleNamespace(
        a_matrix=np.array([[0, 0.4, 0], [0.4, 0, 0.6], [0, 0.6, 0]], dtype=float)
    )
    corrected = rdv.network_from_admm(["a", "b", "c"], corrected_admm, hyper)
    legacy = aa.BenisseNetworkResult(
        node_ids=["a", "b", "c"],
        row=np.array([0]), col=np.array([1]), weight=np.array([0.5]), directed=False,
    )
    legacy_latent = np.array([[0, 1, 3], [1, 0, 2], [3, 2, 0]], dtype=float)
    corrected_latent = np.array([[0, 1.1, 2.8], [1.1, 0, 2.1], [2.8, 2.1, 0]])
    report = rdv.compare_networks(legacy, corrected, legacy_latent, corrected_latent)

    assert corrected.n_edges == 2
    assert report["intersection_edges"] == 1
    assert report["added_edges"] == 1
    assert report["removed_edges"] == 0
    assert report["edge_jaccard"] == 0.5
    assert report["corrected_components"]["largest"] == 3
    assert np.isfinite(report["latent_spearman"])


def test_dense_memory_estimate():
    assert rdv.dense_matrix_mebibytes(203) == pytest.approx(0.3144, rel=1e-3)
    assert rdv.dense_matrix_mebibytes(4833) > 175


def test_validation_rejects_nonconverged_python_result():
    result = ADMMResult(
        q_matrix=np.eye(2), r_matrix=np.zeros((2, 2)), a_matrix=np.zeros((2, 2)),
        sparse_graph=np.zeros((2, 2), dtype=bool), iterations=3, converged=False,
        graph_change_mean=None, graph_change_sd=None, optimizer_results=(),
    )
    with pytest.raises(RuntimeError, match="did not converge"):
        rdv.require_valid_python_result(result, np.zeros((2, 2)))


@pytest.mark.local_data
@pytest.mark.skipif(
    os.environ.get("BENISSE_RUN_LOCAL_DATA_TESTS") != "1",
    reason="set BENISSE_RUN_LOCAL_DATA_TESTS=1 for local AP4 validation",
)
@pytest.mark.skipif(not AP4_PATH.exists(), reason="gitignored AP4 fixture not present")
@pytest.mark.skipif(shutil.which("Rscript") is None, reason="Rscript not available")
def test_ap4_complete_sample_validation(tmp_path):
    report = rdv.validate_mudata_sample(AP4_PATH, tmp_path / "ap4", max_nodes=250)
    assert report["selected_cells"] == 203
    assert report["initialized_nodes"] == 203
    assert report["corrected_python"]["converged"]
    assert report["corrected_python"]["all_optimizers_succeeded"]
    assert report["corrected_python"]["q_min_eigenvalue"] > 0
    assert np.isfinite(report["comparison"]["edge_jaccard"])
    assert np.isfinite(report["comparison"]["latent_spearman"])


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("BENISSE_RUN_PYTHON_CORE_EXAMPLE") != "1",
    reason="set BENISSE_RUN_PYTHON_CORE_EXAMPLE=1 for the paper-example milestone",
)
@pytest.mark.skipif(shutil.which("Rscript") is None, reason="Rscript not available")
def test_corrected_core_on_committed_nsclc_example(tmp_path):
    report = rdv.validate_standard_csv_case(
        EXAMPLE / "10x_NSCLC_exp.csv",
        EXAMPLE / "10x_NSCLC_contigs.csv",
        EXAMPLE / "encoded_10x_NSCLC.csv",
        EXAMPLE,
        tmp_path / "nsclc",
        max_nodes=1500,
    )
    assert report["initialized_nodes"] == 1494
    assert report["corrected_python"]["converged"]
    assert report["corrected_python"]["all_optimizers_succeeded"]
    assert report["corrected_python"]["q_min_eigenvalue"] > 0
    assert np.isfinite(report["comparison"]["edge_jaccard"])
    assert np.isfinite(report["comparison"]["latent_spearman"])
