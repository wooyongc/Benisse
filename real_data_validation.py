"""Guarded local real-data validation for the experimental Python core.

This is internal validation tooling, not a public data-ingress API. It keeps the
Phase 4c R bridge as the supported path and refuses large MuData samples by
default. Stephenson inputs and generated outputs remain local/gitignored.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.stats import spearmanr

import airr_adapter as aa
import benisse_bridge as bridge
from benisse_core import HyperParameters, latent_distances, run_admm


REPO_ROOT = Path(__file__).resolve().parent
INITIALIZER_R = REPO_ROOT / "tests" / "fixtures" / "export_initialized_core.R"
DEFAULT_MAX_NODES = 500


@dataclass(frozen=True)
class PreparedValidationInputs:
    expression_csv: Path
    contigs_csv: Path
    encoder_csv: Path
    selected_cells: int
    estimated_nodes: int
    expression_source: str
    sample_id: str | None


@dataclass(frozen=True)
class InitializedCoreInputs:
    phi: np.ndarray
    si_matrix: np.ndarray
    ls_matrix: np.ndarray
    node_ids: list[str]


def dense_matrix_mebibytes(n_nodes: int) -> float:
    return n_nodes * n_nodes * np.dtype("float64").itemsize / 1024**2


def _expression_matrix(gex, positions, expression_layer):
    if expression_layer == "X":
        matrix = gex.X[positions]
    else:
        if expression_layer not in gex.layers:
            raise ValueError(f"gex layer does not exist: {expression_layer!r}")
        matrix = gex.layers[expression_layer][positions]
    return matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)


def prepare_mudata_validation_inputs(
    h5mu_path,
    output_dir,
    *,
    sample_id=None,
    expression_layer="X",
    max_nodes=DEFAULT_MAX_NODES,
):
    """Prepare one complete MuData sample as standard Benisse CSV inputs."""
    import mudata

    h5mu_path = Path(h5mu_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mdata = mudata.read_h5mu(h5mu_path)
    gex = mdata.mod["gex"]
    heavy = aa.select_heavy_chains(mdata)
    if sample_id is not None:
        if "sample_id" not in gex.obs:
            raise ValueError("gex.obs has no sample_id column")
        sample_cells = set(gex.obs_names[gex.obs["sample_id"].astype(str) == str(sample_id)])
        if not sample_cells:
            raise ValueError(f"sample_id not found: {sample_id!r}")
        heavy = heavy.loc[heavy.index.isin(sample_cells)]
    if heavy.empty:
        raise ValueError("selected sample has no productive heavy chains")
    aa.assert_reversible_barcodes(heavy.index)

    clone_keys = heavy[["v_call", "junction_aa", "j_call"]].fillna("").astype(str).agg(
        "_".join, axis=1
    )
    estimated_nodes = int(clone_keys.nunique())
    if estimated_nodes > max_nodes:
        raise ValueError(
            f"refusing {estimated_nodes} clone nodes; max_nodes={max_nodes}. "
            "Select one biological sample or explicitly raise the guard."
        )

    positions = gex.obs_names.get_indexer(heavy.index)
    if np.any(positions < 0):
        raise ValueError("selected AIRR barcodes are not aligned to the gex modality")
    expression = _expression_matrix(gex, positions, expression_layer)
    expression_frame = pd.DataFrame(
        expression.T,
        index=gex.var_names.astype(str),
        columns=heavy.index.astype(str),
    )
    expression_csv = output_dir / "expression.csv"
    expression_frame.to_csv(expression_csv, index_label="")

    count = pd.to_numeric(heavy["selection_count"], errors="coerce").fillna(1)
    consensus = pd.to_numeric(heavy.get("consensus_count", count), errors="coerce").fillna(count)
    junction_nt = heavy["junction"].fillna("").astype(str)
    contigs = pd.DataFrame(
        {
            "barcode": heavy.index.astype(str),
            "is_cell": "True",
            "contig_id": heavy["sequence_id"].astype(str).to_numpy(),
            "high_confidence": "True",
            "length": junction_nt.str.len().to_numpy(),
            "chain": heavy["locus"].astype(str).to_numpy(),
            "v_gene": heavy["v_call"].fillna("").astype(str).to_numpy(),
            "d_gene": heavy["d_call"].fillna("").astype(str).to_numpy(),
            "j_gene": heavy["j_call"].fillna("").astype(str).to_numpy(),
            "c_gene": heavy["c_call"].fillna("").astype(str).to_numpy(),
            "full_length": "True",
            "productive": "True",
            "cdr3": heavy["junction_aa"].astype(str).to_numpy(),
            "cdr3_nt": junction_nt.to_numpy(),
            "reads": consensus.to_numpy(),
            "umis": count.to_numpy(),
            "raw_clonotype_id": "",
            "raw_consensus_id": "",
        }
    )
    contigs_csv = output_dir / "contigs.csv"
    contigs.to_csv(contigs_csv, index=False)
    encoder_csv = output_dir / "encoder_input.csv"
    aa.write_encoder_input_csv(heavy, encoder_csv)

    metadata = {
        "source": str(h5mu_path),
        "sample_id": sample_id,
        "selected_cells": len(heavy),
        "estimated_nodes": estimated_nodes,
        "expression_source": expression_layer,
        "dense_matrix_mib": dense_matrix_mebibytes(estimated_nodes),
        "validation_only_flags": {
            "is_cell": True,
            "high_confidence": True,
            "full_length": True,
            "productive": True,
        },
    }
    (output_dir / "input_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return PreparedValidationInputs(
        expression_csv=expression_csv,
        contigs_csv=contigs_csv,
        encoder_csv=encoder_csv,
        selected_cells=len(heavy),
        estimated_nodes=estimated_nodes,
        expression_source=expression_layer,
        sample_id=sample_id,
    )


def export_initialized_core(prepared, encoded_csv, output_dir, hyper, *, rscript="Rscript"):
    """Use the production R preparation functions to export exact core matrices."""
    if shutil.which(rscript) is None:
        raise FileNotFoundError(f"Rscript executable not found: {rscript!r}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        rscript,
        str(INITIALIZER_R),
        str(prepared.expression_csv),
        str(prepared.contigs_csv),
        str(encoded_csv),
        str(output_dir),
        str(hyper.lambda1),
        str(hyper.lambda2),
        str(hyper.gamma),
        str(hyper.rho),
        str(hyper.m),
    ]
    subprocess.run(command, cwd=REPO_ROOT, check=True, timeout=300)
    n_nodes = int((output_dir / "n.txt").read_text())

    def read_matrix(name):
        values = np.fromfile(output_dir / name, dtype=np.float64)
        if values.size != n_nodes**2:
            raise ValueError(f"{name} has {values.size} values; expected {n_nodes**2}")
        return values.reshape((n_nodes, n_nodes), order="F")

    node_ids = (output_dir / "node_ids.txt").read_text().splitlines()
    if len(node_ids) != n_nodes or len(set(node_ids)) != n_nodes:
        raise ValueError("exported node identifiers are missing or non-unique")
    return InitializedCoreInputs(
        phi=read_matrix("phi.bin"),
        si_matrix=read_matrix("si.bin"),
        ls_matrix=read_matrix("ls.bin"),
        node_ids=node_ids,
    )


def network_from_admm(node_ids, result, hyper, provenance=None):
    upper = np.triu(result.a_matrix, k=1)
    row, col = np.nonzero(upper > 0)
    network = aa.BenisseNetworkResult(
        node_ids=list(node_ids),
        row=row.astype(np.int64),
        col=col.astype(np.int64),
        weight=upper[row, col].astype(np.float64),
        params=asdict(hyper),
        provenance=aa.benisse_provenance(**(provenance or {})),
        directed=False,
    )
    aa.validate_network_result(network)
    return network


def require_valid_python_result(result, si_matrix):
    if not result.converged:
        raise RuntimeError(
            f"corrected Python core did not converge in {result.iterations} iterations"
        )
    if not result.optimizer_results or not all(
        item.success for item in result.optimizer_results
    ):
        raise RuntimeError("corrected Python core contains a failed optimizer result")
    for matrix, name in (
        (result.a_matrix, "A"),
        (result.q_matrix, "Q"),
        (result.r_matrix, "R"),
    ):
        if not np.isfinite(matrix).all():
            raise RuntimeError(f"corrected Python {name} contains non-finite values")
        if not np.allclose(matrix, matrix.T, rtol=0, atol=1e-10):
            raise RuntimeError(f"corrected Python {name} is not symmetric")
    if not np.allclose(np.diag(result.a_matrix), 0, rtol=0, atol=1e-12):
        raise RuntimeError("corrected Python A has a nonzero diagonal")
    if np.any(result.a_matrix < 0) or np.any(result.a_matrix[si_matrix == 0] != 0):
        raise RuntimeError("corrected Python A violates bounds or crude-graph support")
    if np.linalg.eigvalsh(result.q_matrix).min() <= 0:
        raise RuntimeError("corrected Python Q is not positive definite")


def _component_summary(network):
    n = network.n_nodes
    if n == 0:
        return {"count": 0, "largest": 0, "non_singleton": 0}
    adjacency = sparse.coo_matrix(
        (np.ones(network.n_edges * 2),
         (np.r_[network.row, network.col], np.r_[network.col, network.row])),
        shape=(n, n),
    )
    count, labels = connected_components(adjacency, directed=False)
    sizes = np.bincount(labels, minlength=count)
    return {
        "count": int(count),
        "largest": int(sizes.max()),
        "non_singleton": int(np.count_nonzero(sizes > 1)),
    }


def compare_networks(legacy, corrected, legacy_latent, corrected_latent):
    aa.validate_network_result(legacy)
    aa.validate_network_result(corrected)
    if legacy.node_ids != corrected.node_ids:
        raise ValueError("legacy and corrected node ordering differs")
    expected_shape = (legacy.n_nodes, legacy.n_nodes)
    if np.shape(legacy_latent) != expected_shape or np.shape(corrected_latent) != expected_shape:
        raise ValueError("latent-distance matrices do not match the network dimensions")
    if not np.isfinite(legacy_latent).all() or not np.isfinite(corrected_latent).all():
        raise ValueError("latent-distance matrices contain non-finite values")
    legacy_edges = set(zip(legacy.row.tolist(), legacy.col.tolist()))
    corrected_edges = set(zip(corrected.row.tolist(), corrected.col.tolist()))
    intersection = legacy_edges & corrected_edges
    union = legacy_edges | corrected_edges
    legacy_weights = dict(zip(zip(legacy.row, legacy.col), legacy.weight))
    corrected_weights = dict(zip(zip(corrected.row, corrected.col), corrected.weight))
    common_weight_delta = np.asarray(
        [corrected_weights[edge] - legacy_weights[edge] for edge in sorted(intersection)]
    )
    triangle = np.triu_indices(legacy.n_nodes, k=1)
    legacy_vector = np.asarray(legacy_latent)[triangle]
    corrected_vector = np.asarray(corrected_latent)[triangle]
    latent_correlation = spearmanr(legacy_vector, corrected_vector).statistic
    return {
        "n_nodes": legacy.n_nodes,
        "legacy_edges": legacy.n_edges,
        "corrected_edges": corrected.n_edges,
        "intersection_edges": len(intersection),
        "added_edges": len(corrected_edges - legacy_edges),
        "removed_edges": len(legacy_edges - corrected_edges),
        "edge_jaccard": len(intersection) / len(union) if union else 1.0,
        "common_weight_rmse": float(np.sqrt(np.mean(common_weight_delta**2)))
        if common_weight_delta.size else None,
        "common_weight_max_abs": float(np.max(np.abs(common_weight_delta)))
        if common_weight_delta.size else None,
        "legacy_components": _component_summary(legacy),
        "corrected_components": _component_summary(corrected),
        "latent_spearman": float(latent_correlation),
        "latent_rmse": float(np.sqrt(np.mean((corrected_vector - legacy_vector) ** 2))),
    }


def validate_mudata_sample(
    h5mu_path,
    output_dir,
    *,
    sample_id=None,
    expression_layer="X",
    max_nodes=DEFAULT_MAX_NODES,
    max_iterations=100,
    timeout=900,
):
    output_dir = Path(output_dir)
    prepared = prepare_mudata_validation_inputs(
        h5mu_path,
        output_dir / "inputs",
        sample_id=sample_id,
        expression_layer=expression_layer,
        max_nodes=max_nodes,
    )
    params = bridge.BenisseRParams(lambda2=prepared.selected_cells, max_iter=max_iterations)
    hyper = HyperParameters(
        lambda1=params.lambda1,
        lambda2=params.lambda2,
        gamma=params.gamma,
        rho=params.rho,
        m=params.m,
    )
    encoded_csv = output_dir / "inputs" / "encoded.csv"
    encoder_start = time.monotonic()
    bridge.encode_bcr_csv(prepared.encoder_csv, encoded_csv, cuda=False)
    encoder_seconds = time.monotonic() - encoder_start
    initialized = export_initialized_core(
        prepared, encoded_csv, output_dir / "initialized", hyper
    )
    if len(initialized.node_ids) > max_nodes:
        raise ValueError("R initialization exceeded the node guard")

    python_start = time.monotonic()
    python_result = run_admm(
        initialized.phi,
        initialized.si_matrix,
        initialized.ls_matrix,
        hyper,
        max_iterations=max_iterations,
        stop_cutoff=params.stop_cutoff,
    )
    require_valid_python_result(python_result, initialized.si_matrix)
    python_seconds = time.monotonic() - python_start
    corrected = network_from_admm(
        initialized.node_ids,
        python_result,
        hyper,
        provenance={"implementation": "experimental_corrected_python"},
    )

    legacy_dir = output_dir / "legacy_r"
    legacy_start = time.monotonic()
    bridge.run_benisse_r(
        prepared.expression_csv,
        prepared.contigs_csv,
        encoded_csv,
        legacy_dir,
        params,
        timeout=timeout,
    )
    legacy_seconds = time.monotonic() - legacy_start
    legacy = bridge.read_benisse_network(legacy_dir, params=params)
    legacy_latent = np.atleast_2d(np.loadtxt(legacy_dir / "latent_dist.txt"))
    corrected_latent = latent_distances(python_result.q_matrix, hyper.m, hyper.gamma)
    comparison = compare_networks(legacy, corrected, legacy_latent, corrected_latent)
    report = {
        "scope": "local_validation_only",
        "dataset": str(h5mu_path),
        "sample_id": sample_id,
        "expression_source": expression_layer,
        "selected_cells": prepared.selected_cells,
        "estimated_nodes": prepared.estimated_nodes,
        "initialized_nodes": len(initialized.node_ids),
        "dense_matrix_mib": dense_matrix_mebibytes(len(initialized.node_ids)),
        "params": asdict(params),
        "timing_seconds": {
            "encoder": encoder_seconds,
            "corrected_python_core": python_seconds,
            "legacy_r_pipeline": legacy_seconds,
        },
        "corrected_python": {
            "converged": python_result.converged,
            "iterations": python_result.iterations,
            "all_optimizers_succeeded": all(
                item.success for item in python_result.optimizer_results
            ),
            "q_min_eigenvalue": float(np.linalg.eigvalsh(python_result.q_matrix).min()),
        },
        "comparison": comparison,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def validate_standard_csv_case(
    expression_csv,
    contigs_csv,
    encoded_csv,
    legacy_output_dir,
    output_dir,
    *,
    params=None,
    max_nodes=2000,
):
    """Validate the corrected core against existing standard CSV/R outputs.

    This is the opt-in paper-example path. It reuses an existing encoded CSV and
    legacy result directory, so it does not rerun the encoder or legacy R model.
    """
    params = params or bridge.BenisseRParams()
    hyper = HyperParameters(
        lambda1=params.lambda1,
        lambda2=params.lambda2,
        gamma=params.gamma,
        rho=params.rho,
        m=params.m,
    )
    output_dir = Path(output_dir)
    prepared = PreparedValidationInputs(
        expression_csv=Path(expression_csv),
        contigs_csv=Path(contigs_csv),
        encoder_csv=Path(encoded_csv),
        selected_cells=0,
        estimated_nodes=0,
        expression_source="existing_standard_csv",
        sample_id=None,
    )
    initialized = export_initialized_core(
        prepared, encoded_csv, output_dir / "initialized", hyper
    )
    if len(initialized.node_ids) > max_nodes:
        raise ValueError(
            f"refusing {len(initialized.node_ids)} initialized nodes; max_nodes={max_nodes}"
        )
    python_start = time.monotonic()
    python_result = run_admm(
        initialized.phi,
        initialized.si_matrix,
        initialized.ls_matrix,
        hyper,
        max_iterations=params.max_iter,
        stop_cutoff=params.stop_cutoff,
    )
    require_valid_python_result(python_result, initialized.si_matrix)
    python_seconds = time.monotonic() - python_start
    corrected = network_from_admm(
        initialized.node_ids,
        python_result,
        hyper,
        provenance={"implementation": "experimental_corrected_python"},
    )
    legacy_output_dir = Path(legacy_output_dir)
    legacy = bridge.read_benisse_network(legacy_output_dir, params=params)
    legacy_latent = np.atleast_2d(np.loadtxt(legacy_output_dir / "latent_dist.txt"))
    corrected_latent = latent_distances(python_result.q_matrix, hyper.m, hyper.gamma)
    report = {
        "scope": "opt_in_paper_example_validation",
        "initialized_nodes": len(initialized.node_ids),
        "dense_matrix_mib": dense_matrix_mebibytes(len(initialized.node_ids)),
        "params": asdict(params),
        "timing_seconds": {"corrected_python_core": python_seconds},
        "corrected_python": {
            "converged": python_result.converged,
            "iterations": python_result.iterations,
            "all_optimizers_succeeded": all(
                item.success for item in python_result.optimizer_results
            ),
            "q_min_eigenvalue": float(np.linalg.eigvalsh(python_result.q_matrix).min()),
        },
        "comparison": compare_networks(
            legacy, corrected, legacy_latent, corrected_latent
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("h5mu")
    parser.add_argument("output_dir")
    parser.add_argument("--sample-id")
    parser.add_argument("--expression-layer", default="X")
    parser.add_argument("--max-nodes", type=int, default=DEFAULT_MAX_NODES)
    parser.add_argument("--max-iterations", type=int, default=100)
    args = parser.parse_args(argv)
    report = validate_mudata_sample(
        args.h5mu,
        args.output_dir,
        sample_id=args.sample_id,
        expression_layer=args.expression_layer,
        max_nodes=args.max_nodes,
        max_iterations=args.max_iterations,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
