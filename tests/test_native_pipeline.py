"""Parity and integration tests for the R-free corrected v2 pipeline."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
import pytest
from scipy import sparse


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import airr_adapter as aa  # noqa: E402
import benisse_bridge as bridge  # noqa: E402
import benisse_pipeline as pipeline  # noqa: E402
import benisse_preprocessing as prep  # noqa: E402
from benisse_core import HyperParameters  # noqa: E402


EXAMPLE = REPO_ROOT / "example"
EXPORT_R = REPO_ROOT / "tests" / "fixtures" / "export_initialized_core.R"
RSCRIPT = shutil.which("Rscript")
anndata = pytest.importorskip("anndata")
mudata = pytest.importorskip("mudata")


def _encoded(cdr3_values):
    rows = []
    for index, cdr3 in enumerate(cdr3_values):
        row = {str(i): float(index + i / 10) for i in range(20)}
        row["index"] = cdr3
        rows.append(row)
    return pd.DataFrame(rows)


def _small_standard_frames():
    cells = ["cell-A", "cell-B", "cell-C", "cell-D", "cell-E", "cell-F"]
    base = np.array([0, 2, 1, 5, 3, 8], dtype=float)
    expression = pd.DataFrame(
        np.vstack([(index + 1) * base + (index % 3) for index in range(12)]),
        index=[f"gene_{i}" for i in range(12)],
        columns=cells,
    )
    rows = []
    clone_rows = [
        ("cell-A", "V1", "J1", "CA", 2),
        ("cell-B", "V1", "J1", "CB", 4),
        ("cell-C", "V1", "J1", "CA", 3),
        ("cell-D", "V2", "J2", "CC", 5),
        ("cell-E", "V2", "J2", "CD", 6),
        ("cell-F", "V2", "J2", "CD", 2),
    ]
    for barcode, v_gene, j_gene, cdr3, umis in clone_rows:
        rows.append(
            {
                "barcode": barcode,
                "is_cell": True,
                "high_confidence": True,
                "full_length": True,
                "productive": True,
                "chain": "IGH",
                "v_gene": v_gene,
                "j_gene": j_gene,
                "cdr3": cdr3,
                "cdr3_nt": f"NT{cdr3}",
                "umis": umis,
            }
        )
    # A lower-UMI duplicate must lose, and QC/light-chain rows must not enter.
    rows.extend(
        [
            {**rows[1], "v_gene": "WRONG", "umis": 1},
            {**rows[3], "barcode": "cell-X", "is_cell": False},
            {**rows[4], "barcode": "cell-Y", "chain": "IGK"},
        ]
    )
    return expression, pd.DataFrame(rows), _encoded(["CA", "CB", "CC", "CD"])


def test_prepare_frames_preserves_r_order_qc_and_initialization_invariants():
    expression, contigs, encoded = _small_standard_frames()
    hyper = HyperParameters(lambda1=0.8, lambda2=6, gamma=1, rho=1, m=3)
    result = prep.prepare_frames(expression, contigs, encoded, hyper)

    assert result.expression.columns.tolist() == [
        "cell.A", "cell.B", "cell.C", "cell.D", "cell.E", "cell.F"
    ]
    assert result.node_ids == ("V1_CA_J1", "V1_CB_J1", "V2_CC_J2", "V2_CD_J2")
    assert result.cell_clone_ids.tolist() == [
        "V1_CA_J1", "V1_CB_J1", "V1_CA_J1", "V2_CC_J2", "V2_CD_J2", "V2_CD_J2"
    ]
    assert result.annotation["clsize"].tolist() == [2, 1, 1, 2]
    assert result.annotation.iloc[1]["v_gene"] == "V1"  # lower-UMI WRONG row lost
    expected_si = np.array(
        [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=float
    )
    np.testing.assert_array_equal(result.si_matrix, expected_si)
    np.testing.assert_allclose(result.phi, result.phi.T, rtol=0, atol=1e-14)
    np.testing.assert_allclose(np.diag(result.phi), 0, rtol=0, atol=1e-14)
    np.testing.assert_allclose(result.ls_matrix.sum(axis=1), 0, rtol=0, atol=1e-14)
    np.testing.assert_array_equal(result.initial_a, hyper.lambda1 * expected_si)


def test_prepare_frames_rejects_missing_contract_and_handles_empty_candidate_graph():
    expression, contigs, encoded = _small_standard_frames()
    hyper = HyperParameters(lambda1=1, lambda2=6, gamma=1, rho=1, m=2)
    with pytest.raises(ValueError, match="missing required columns"):
        prep.prepare_frames(expression, contigs.drop(columns="umis"), encoded, hyper)

    contigs.loc[:, "v_gene"] = [f"V{i}" for i in range(len(contigs))]
    result = prep.prepare_frames(expression, contigs, encoded, hyper)
    assert not result.si_matrix.any()
    assert not result.ls_matrix.any()

    annotation = result.annotation.copy()
    annotation.iloc[0, annotation.columns.get_loc("v_gene")] = ""
    with pytest.raises(ValueError, match="missing V/J calls"):
        prep.initialize_core_inputs(
            result.expression,
            result.cell_clone_ids,
            annotation,
            result.embedding,
            hyper,
        )


def test_standard_csv_pipeline_import_does_not_require_awkward():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.modules['awkward'] = None; import benisse_pipeline; print('ok')",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "ok"


@pytest.mark.skipif(RSCRIPT is None, reason="Rscript is required for frozen-preprocessing parity")
def test_nsclc_preprocessing_matches_frozen_r_oracle(tmp_path):
    subprocess.run(
        [
            RSCRIPT,
            str(EXPORT_R),
            str(EXAMPLE / "10x_NSCLC_exp.csv"),
            str(EXAMPLE / "10x_NSCLC_contigs.csv"),
            str(EXAMPLE / "encoded_10x_NSCLC.csv"),
            str(tmp_path),
            "1", "1610", "1", "1", "10",
        ],
        cwd=REPO_ROOT,
        check=True,
        timeout=180,
    )
    hyper = HyperParameters(lambda1=1, lambda2=1610, gamma=1, rho=1, m=10)
    result = prep.prepare_csv_inputs(
        EXAMPLE / "10x_NSCLC_exp.csv",
        EXAMPLE / "10x_NSCLC_contigs.csv",
        EXAMPLE / "encoded_10x_NSCLC.csv",
        hyper,
    )
    assert result.n_cells == 1612
    assert result.n_nodes == 1494
    assert list(result.node_ids) == (tmp_path / "node_ids.txt").read_text().splitlines()
    assert result.cell_clone_ids.tolist() == (tmp_path / "cell_clone_ids.txt").read_text().splitlines()
    assert result.expression.columns.tolist() == (tmp_path / "selected_barcodes.txt").read_text().splitlines()

    def read_square(name):
        return np.fromfile(tmp_path / name, dtype=np.float64).reshape(
            (result.n_nodes, result.n_nodes), order="F"
        )

    np.testing.assert_allclose(result.phi, read_square("phi.bin"), rtol=0, atol=2e-14)
    np.testing.assert_array_equal(result.si_matrix, read_square("si.bin"))
    np.testing.assert_allclose(result.ls_matrix, read_square("ls.bin"), rtol=2e-12, atol=2e-18)
    np.testing.assert_allclose(
        result.master_dist_e,
        read_square("master_dist_e.bin"),
        rtol=2e-6,
        atol=2e-8,
    )
    r_embedding = np.fromfile(tmp_path / "embedding.bin", dtype=np.float64).reshape(
        (result.n_nodes, 20), order="F"
    )
    np.testing.assert_allclose(result.embedding, r_embedding, rtol=0, atol=5e-16)
    r_expression = np.fromfile(tmp_path / "expression.bin", dtype=np.float64).reshape(
        result.expression.shape, order="F"
    )
    np.testing.assert_allclose(result.expression.to_numpy(), r_expression, rtol=0, atol=0)


def test_corrected_pipeline_writes_parseable_outputs_and_provenance(tmp_path):
    fixture = json.loads((REPO_ROOT / "tests" / "fixtures" / "r_core_golden.json").read_text())
    inputs = {name: np.asarray(value, dtype=float) for name, value in fixture["small"]["inputs"].items()}
    hyper = HyperParameters(**fixture["hyperparameters"])
    node_ids = ("V1_C1_J1", "V1_C2_J1", "V1_C3_J1", "V2_C4_J2")
    annotation = pd.DataFrame(
        {
            "v_gene": ["V1", "V1", "V1", "V2"],
            "j_gene": ["J1", "J1", "J1", "J2"],
            "cdr3": ["C1", "C2", "C3", "C4"],
            "barcode": ["b1", "b2", "b3", "b4"],
            "clsize": [1, 1, 1, 1],
        },
        index=pd.Index(node_ids, name="clone"),
    )
    embedding = np.pad(np.array([[0, 0], [1, 0], [0, 2], [2, 2]], dtype=float), ((0, 0), (0, 18)))
    prepared = prep.PreparedCoreInputs(
        expression=pd.DataFrame(np.eye(4), columns=["b1", "b2", "b3", "b4"]),
        cell_clone_ids=np.asarray(node_ids),
        node_ids=node_ids,
        annotation=annotation,
        embedding=embedding,
        master_dist_e=np.zeros((4, 4)),
        phi=inputs["phi"],
        si_matrix=inputs["SI"],
        identity=inputs["I"],
        initial_a=inputs["A"],
        ls_matrix=inputs["LS"],
    )
    params = pipeline.BenisseParams(
        lambda1=hyper.lambda1,
        lambda2=hyper.lambda2,
        gamma=hyper.gamma,
        rho=hyper.rho,
        m=hyper.m,
        max_iterations=30,
    )
    result = pipeline.run_prepared_pipeline(
        prepared, tmp_path, params=params, generate_plots=False
    )
    assert result.core.converged and result.core.iterations == 23
    assert result.network.provenance["implementation"] == "corrected_python_v2"
    assert result.network.provenance["runtime_requires_r"] is False
    assert len(result.network.provenance["scientific_inputs_sha256"]) == 64
    assert result.network.provenance["software_versions"]["numpy"] == np.__version__
    assert result.network.provenance["code"]["revision"]
    parsed = bridge.read_benisse_network(tmp_path)
    assert parsed.node_ids == list(node_ids)
    assert set(zip(parsed.row, parsed.col)) == set(
        zip(result.network.row, result.network.col)
    )
    metadata = json.loads((tmp_path / "python_pipeline_metadata.json").read_text())
    assert metadata["runtime_requires_r"] is False
    assert metadata["n_edges"] == result.network.n_edges


def _chain(sequence_id, junction_aa, v_call, j_call, count):
    return {
        "locus": "IGH",
        "productive": True,
        "junction_aa": junction_aa,
        "junction": "TGTGCCGCCTGG",
        "sequence_id": sequence_id,
        "v_call": v_call,
        "d_call": "IGHD1",
        "j_call": j_call,
        "c_call": "IGHM",
        "duplicate_count": count,
        "consensus_count": count * 10,
    }


def test_mudata_pipeline_is_r_free_and_attaches_result(tmp_path, monkeypatch):
    cells = ["cell_A-1", "cell_B-1", "cell_C-1", "cell_D-1"]
    airr = anndata.AnnData(np.zeros((4, 0), dtype=np.float32))
    airr.obs_names = cells
    airr.obsm["airr"] = ak.Array(
        [
            [_chain("a", "CARAAW", "V1", "J1", 5)],
            [_chain("b", "CARBBW", "V2", "J2", 4)],
            [_chain("c", "CARCCW", "V3", "J3", 3)],
            [_chain("d", "CARAAW", "V1", "J1", 2)],
        ]
    )
    values = np.random.default_rng(7).integers(0, 20, size=(4, 12))
    gex = anndata.AnnData(sparse.csr_matrix(values))
    gex.obs_names = cells
    gex.var_names = [f"g{i}" for i in range(12)]
    mdata = mudata.MuData({"gex": gex, "airr": airr})

    def fail_if_r_called(*args, **kwargs):
        raise AssertionError("R runtime must not be invoked")

    monkeypatch.setattr(bridge, "run_benisse_r", fail_if_r_called)
    attached, result = pipeline.run_mudata_pipeline(
        mdata,
        tmp_path,
        params=pipeline.BenisseParams(lambda2=4, m=2),
        generate_plots=False,
        max_nodes=5,
    )
    stored = aa.read_network_result(attached)
    assert stored.node_ids == result.network.node_ids
    assert stored.provenance["runtime_requires_r"] is False
    assert len(stored.provenance["encoder_model"]["sha256"]) == 64
    assert set(stored.provenance["inputs_sha256"]) == {
        "encoder_input_csv", "encoded_csv"
    }
    assert attached.mod["airr"].obsm["X_benisse"].shape == (4, 20)
    clone_ids = attached.mod["airr"].obs["benisse_clone_id"].astype(str).tolist()
    assert clone_ids[0] == clone_ids[3] == "V1_CARAAW_J1"
    assert result.prepared.n_cells == 4
    assert result.prepared.n_nodes == 3

    h5mu = tmp_path / "result.h5mu"
    attached.write_h5mu(h5mu)
    restored = mudata.read_h5mu(h5mu)
    assert restored.mod["airr"].obs["benisse_clone_id"].astype(str).tolist() == clone_ids
    assert aa.read_network_result(restored).node_ids == stored.node_ids


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("BENISSE_RUN_NATIVE_EXAMPLE") != "1",
    reason="set BENISSE_RUN_NATIVE_EXAMPLE=1 for the R-free NSCLC milestone",
)
def test_native_nsclc_end_to_end_changed_v2_expectations(tmp_path):
    result = pipeline.run_csv_pipeline(
        EXAMPLE / "10x_NSCLC.csv",
        EXAMPLE / "10x_NSCLC_exp.csv",
        EXAMPLE / "10x_NSCLC_contigs.csv",
        tmp_path,
        params=pipeline.BenisseParams(),
        generate_plots=True,
        max_nodes=1500,
    )
    assert result.prepared.n_cells == 1612
    assert result.prepared.n_nodes == 1494
    assert result.network.n_edges == 1592  # corrected v2; frozen v1 R has 1691
    assert result.core.iterations == 32
    assert len(result.plot_paths) == 4
    assert result.network.provenance["runtime_requires_r"] is False


@pytest.mark.local_data
@pytest.mark.parametrize(
    ("sample_id", "cells", "nodes", "edges", "iterations"),
    [
        ("BGCV09_CV0171", 94, 92, 11, 20),
        ("AP4", 203, 203, 22, 19),
        ("MH9143277", 437, 427, 98, 23),
    ],
)
@pytest.mark.skipif(
    os.environ.get("BENISSE_RUN_NATIVE_LOCAL_DATA") != "1",
    reason="set BENISSE_RUN_NATIVE_LOCAL_DATA=1 for R-free Stephenson validation",
)
def test_native_stephenson_samples(
    tmp_path, sample_id, cells, nodes, edges, iterations
):
    source = REPO_ROOT / "data" / "external" / "stephenson2021_5k.h5mu"
    if not source.exists():
        pytest.skip("gitignored Stephenson MuData object is not present")
    _, result = pipeline.run_mudata_pipeline(
        source,
        tmp_path / sample_id,
        sample_id=sample_id,
        generate_plots=True,
        max_nodes=500,
    )
    assert (result.prepared.n_cells, result.prepared.n_nodes) == (cells, nodes)
    assert (result.network.n_edges, result.core.iterations) == (edges, iterations)
    assert len(result.plot_paths) == 4
    assert result.network.provenance["runtime_requires_r"] is False
