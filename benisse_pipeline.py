"""R-free end-to-end Benisse v2 pipeline.

The corrected Python numerical core is the v2 default in this module.  The
historical R implementation remains a frozen validation oracle and is never
invoked by the runtime functions below.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components

import airr_adapter as aa
from benisse_core import ADMMResult, HyperParameters, latent_distances, run_admm
from benisse_preprocessing import PreparedCoreInputs, prepare_csv_inputs, prepare_frames


@dataclass(frozen=True)
class BenisseParams:
    """Scientific and controller parameters for the corrected v2 pipeline."""

    lambda2: float = 1610
    gamma: float = 1
    max_iterations: int = 100
    lambda1: float = 1
    rho: float = 1
    m: int = 10
    stop_cutoff: float = 1e-10
    edge_tolerance: float = 0.0

    def hyperparameters(self) -> HyperParameters:
        return HyperParameters(
            lambda1=self.lambda1,
            lambda2=self.lambda2,
            gamma=self.gamma,
            rho=self.rho,
            m=self.m,
        )


@dataclass(frozen=True)
class NativePipelineResult:
    """In-memory outputs from the R-free corrected pipeline."""

    network: aa.BenisseNetworkResult
    prepared: PreparedCoreInputs
    core: ADMMResult
    latent_distances: np.ndarray
    annotation: pd.DataFrame
    plot_paths: dict[str, Path]
    output_paths: dict[str, Path]


def encode_bcr_csv(encoder_input_csv, encoded_csv, *, cuda=False) -> Path:
    """Run the existing Python encoder without importing the legacy R bridge."""
    from AchillesEncoder import encode_bcr

    encoded_csv = Path(encoded_csv)
    encode_bcr(str(encoder_input_csv), str(encoded_csv), cuda=cuda)
    return encoded_csv


def _validate_params(params: BenisseParams) -> None:
    params.hyperparameters()  # benisse_core validates again at execution
    if not isinstance(params.max_iterations, int) or params.max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    if not np.isfinite(params.stop_cutoff) or params.stop_cutoff < 0:
        raise ValueError("stop_cutoff must be finite and non-negative")
    if not np.isfinite(params.edge_tolerance) or params.edge_tolerance < 0:
        raise ValueError("edge_tolerance must be finite and non-negative")


def _validate_core_result(result: ADMMResult, si_matrix: np.ndarray) -> None:
    if not result.converged:
        raise RuntimeError(f"corrected Python core did not converge in {result.iterations} iterations")
    if not result.optimizer_results or not all(item.success for item in result.optimizer_results):
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
        raise RuntimeError("corrected Python A violates bounds or V/J candidate support")
    if np.linalg.eigvalsh(result.q_matrix).min() <= 0:
        raise RuntimeError("corrected Python Q is not positive definite")


def _component_annotation(network: aa.BenisseNetworkResult) -> np.ndarray:
    adjacency = sparse.coo_matrix(
        (
            np.ones(network.n_edges * 2),
            (np.r_[network.row, network.col], np.r_[network.col, network.row]),
        ),
        shape=(network.n_nodes, network.n_nodes),
    )
    count, labels = connected_components(adjacency, directed=False)
    sizes = np.bincount(labels, minlength=count)
    order = sorted(range(count), key=lambda item: (-sizes[item], np.flatnonzero(labels == item)[0]))
    remap = np.empty(count, dtype="int64")
    remap[order] = np.arange(count)
    labels = remap[labels]
    remapped_sizes = np.bincount(labels, minlength=count)
    return np.asarray(
        [
            f"single {index + 1}" if remapped_sizes[label] == 1 else f"cluster {label + 1}"
            for index, label in enumerate(labels)
        ]
    )


def _network_from_core(
    prepared: PreparedCoreInputs,
    core: ADMMResult,
    params: BenisseParams,
    provenance: dict[str, Any] | None = None,
) -> aa.BenisseNetworkResult:
    upper = np.triu(core.a_matrix, k=1)
    row, col = np.nonzero(upper > params.edge_tolerance)
    network = aa.BenisseNetworkResult(
        node_ids=list(prepared.node_ids),
        row=row.astype("int64"),
        col=col.astype("int64"),
        weight=upper[row, col].astype("float64"),
        params=asdict(params),
        provenance=aa.benisse_provenance(
            implementation="corrected_python_v2",
            runtime_requires_r=False,
            edge_tolerance=params.edge_tolerance,
            n_cells=prepared.n_cells,
            n_nodes=prepared.n_nodes,
            **(provenance or {}),
        ),
        directed=False,
    )
    aa.validate_network_result(network)
    return network


def _write_outputs(
    output_dir: Path,
    prepared: PreparedCoreInputs,
    core: ADMMResult,
    latent: np.ndarray,
    network: aa.BenisseNetworkResult,
    annotation: pd.DataFrame,
    params: BenisseParams,
    provenance: dict[str, Any],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "clone_annotation": output_dir / "clone_annotation.csv",
        "cleaned_exp": output_dir / "cleaned_exp.txt",
        "clonality_label": output_dir / "clonality_label.txt",
        "sparse_graph": output_dir / "sparse_graph.txt",
        "latent_dist": output_dir / "latent_dist.txt",
        "metadata": output_dir / "python_pipeline_metadata.json",
    }
    written_annotation = annotation.set_index("barcode")[
        ["v_gene", "j_gene", "cdr3", "clsize", "graph_label"]
    ]
    written_annotation.to_csv(paths["clone_annotation"])
    prepared.expression.to_csv(paths["cleaned_exp"], sep=" ", index=True)
    pd.Series(prepared.cell_clone_ids).to_csv(
        paths["clonality_label"], index=False, header=False
    )
    written_a = core.a_matrix.copy()
    written_a[written_a <= params.edge_tolerance] = 0
    np.savetxt(paths["sparse_graph"], written_a)
    np.savetxt(paths["latent_dist"], latent)
    metadata = {
        "implementation": "corrected_python_v2",
        "runtime_requires_r": False,
        "params": asdict(params),
        "n_cells": prepared.n_cells,
        "n_nodes": prepared.n_nodes,
        "n_edges": network.n_edges,
        "iterations": core.iterations,
        "converged": core.converged,
        "all_optimizers_succeeded": all(item.success for item in core.optimizer_results),
        "q_min_eigenvalue": float(np.linalg.eigvalsh(core.q_matrix).min()),
        "provenance": provenance,
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2) + "\n")
    return paths


def run_prepared_pipeline(
    prepared: PreparedCoreInputs,
    output_dir,
    *,
    params: BenisseParams,
    generate_plots: bool = True,
    provenance: dict[str, Any] | None = None,
) -> NativePipelineResult:
    """Run the corrected core and Python post-analysis without invoking R."""
    _validate_params(params)
    output_dir = Path(output_dir)
    core = run_admm(
        prepared.phi,
        prepared.si_matrix,
        prepared.ls_matrix,
        params.hyperparameters(),
        max_iterations=params.max_iterations,
        stop_cutoff=params.stop_cutoff,
    )
    _validate_core_result(core, prepared.si_matrix)
    latent = latent_distances(core.q_matrix, params.m, params.gamma)
    latent[np.abs(latent) < 1e-12] = 0
    provenance = dict(provenance or {})
    network = _network_from_core(prepared, core, params, provenance)
    annotation = prepared.annotation.copy()
    annotation["graph_label"] = _component_annotation(network)
    output_paths = _write_outputs(
        output_dir, prepared, core, latent, network, annotation, params, provenance
    )
    plot_paths = {}
    if generate_plots:
        from benisse_plotting import BenissePlotData, generate_plots as render_plots

        plot_data = BenissePlotData(network, annotation, latent, prepared.embedding)
        plot_paths = render_plots(
            plot_data,
            output_dir,
            expression_distances=prepared.master_dist_e,
        )
    return NativePipelineResult(
        network=network,
        prepared=prepared,
        core=core,
        latent_distances=latent,
        annotation=annotation,
        plot_paths=plot_paths,
        output_paths=output_paths,
    )


def run_encoded_csv_pipeline(
    expression_csv,
    contigs_csv,
    encoded_csv,
    output_dir,
    *,
    params: BenisseParams | None = None,
    generate_plots: bool = True,
    max_nodes: int = 2000,
    provenance: dict[str, Any] | None = None,
) -> NativePipelineResult:
    """Run v2 from existing standard CSVs and a precomputed encoder output."""
    params = params or BenisseParams()
    prepared = prepare_csv_inputs(
        expression_csv,
        contigs_csv,
        encoded_csv,
        params.hyperparameters(),
    )
    if prepared.n_nodes > max_nodes:
        raise ValueError(f"refusing {prepared.n_nodes} nodes; max_nodes={max_nodes}")
    source_provenance = {
        "source_type": "standard_csv",
        "expression_csv": str(Path(expression_csv)),
        "contigs_csv": str(Path(contigs_csv)),
        "encoded_csv": str(Path(encoded_csv)),
        **(provenance or {}),
    }
    return run_prepared_pipeline(
        prepared,
        output_dir,
        params=params,
        generate_plots=generate_plots,
        provenance=source_provenance,
    )


def run_csv_pipeline(
    encoder_input_csv,
    expression_csv,
    contigs_csv,
    output_dir,
    *,
    params: BenisseParams | None = None,
    cuda: bool = False,
    generate_plots: bool = True,
    max_nodes: int = 2000,
) -> NativePipelineResult:
    """Encode BCRs and run the complete corrected v2 CSV workflow in Python."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    encoded_csv = encode_bcr_csv(
        encoder_input_csv, output_dir / "encoded.csv", cuda=cuda
    )
    return run_encoded_csv_pipeline(
        expression_csv,
        contigs_csv,
        encoded_csv,
        output_dir,
        params=params,
        generate_plots=generate_plots,
        max_nodes=max_nodes,
        provenance={"encoder_input_csv": str(Path(encoder_input_csv))},
    )


def _mudata_frames(
    mdata,
    *,
    sample_id=None,
    expression_layer="X",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
        raise ValueError("selected MuData scope has no productive heavy chains")
    positions = gex.obs_names.get_indexer(heavy.index)
    if np.any(positions < 0):
        raise ValueError("AIRR-selected cells are not aligned to the gex modality")
    if expression_layer == "X":
        matrix = gex.X[positions]
    else:
        if expression_layer not in gex.layers:
            raise ValueError(f"gex layer does not exist: {expression_layer!r}")
        matrix = gex.layers[expression_layer][positions]
    matrix = matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)
    expression = pd.DataFrame(
        matrix.T,
        index=gex.var_names.astype(str),
        columns=heavy.index.astype(str),
    )
    count = pd.to_numeric(heavy["selection_count"], errors="coerce").fillna(1)
    junction_nt = heavy["junction"].fillna("").astype(str)
    contigs = pd.DataFrame(
        {
            "barcode": heavy.index.astype(str),
            "is_cell": True,
            "high_confidence": True,
            "full_length": True,
            "productive": True,
            "chain": heavy["locus"].astype(str).to_numpy(),
            "v_gene": heavy["v_call"].fillna("").astype(str).to_numpy(),
            "j_gene": heavy["j_call"].fillna("").astype(str).to_numpy(),
            "cdr3": heavy["junction_aa"].astype(str).to_numpy(),
            "cdr3_nt": junction_nt.to_numpy(),
            "umis": count.to_numpy(),
        }
    )
    return expression, contigs, heavy


def run_mudata_pipeline(
    obj,
    output_dir,
    *,
    params: BenisseParams | None = None,
    sample_id=None,
    expression_layer="X",
    cuda=False,
    generate_plots=True,
    max_nodes=500,
) -> tuple[Any, NativePipelineResult]:
    """Run the complete R-free workflow from MuData and attach its network result."""
    import mudata

    mdata = mudata.read_h5mu(obj) if isinstance(obj, (str, Path)) else obj
    expression, contigs, heavy = _mudata_frames(
        mdata, sample_id=sample_id, expression_layer=expression_layer
    )
    estimated_nodes = int(
        heavy[["v_call", "junction_aa", "j_call"]]
        .fillna("")
        .astype(str)
        .agg("_".join, axis=1)
        .nunique()
    )
    if estimated_nodes > max_nodes:
        raise ValueError(f"refusing {estimated_nodes} nodes; max_nodes={max_nodes}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    encoder_input = output_dir / "encoder_input.csv"
    aa.write_encoder_input_csv(heavy, encoder_input)
    encoded_csv = encode_bcr_csv(
        encoder_input, output_dir / "encoded.csv", cuda=cuda
    )
    encoded = pd.read_csv(encoded_csv)
    if params is None:
        params = BenisseParams(lambda2=expression.shape[1])
    prepared = prepare_frames(expression, contigs, encoded, params.hyperparameters())
    if prepared.n_nodes > max_nodes:
        raise ValueError(f"refusing {prepared.n_nodes} nodes; max_nodes={max_nodes}")
    source_name = str(obj) if isinstance(obj, (str, Path)) else "in_memory_mudata"
    result = run_prepared_pipeline(
        prepared,
        output_dir,
        params=params,
        generate_plots=generate_plots,
        provenance={
            "source_type": "mudata",
            "source": source_name,
            "sample_id": None if sample_id is None else str(sample_id),
            "expression_layer": expression_layer,
        },
    )
    lookup = encoded.set_index(encoded["index"].astype(str))[list(map(str, range(20)))]
    cell_embedding = lookup.loc[heavy["junction_aa"].astype(str)].copy()
    cell_embedding.index = heavy.index
    aa.attach_embedding(mdata, cell_embedding)
    aa.attach_network_result(mdata, result.network)
    return mdata, result
