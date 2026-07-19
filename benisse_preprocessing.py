"""R-free preparation and initialization for the corrected Benisse v2 core.

This module ports the scientific calculations in ``R/prepare.R`` and
``R/initiation.R``.  It deliberately preserves clone ordering and the legacy
expression-distance normalization so that those inputs can be checked directly
against the frozen R oracle.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA

from benisse_core import HyperParameters


ENCODER_COLUMNS = tuple(str(i) for i in range(20))
REQUIRED_CONTIG_COLUMNS = {
    "barcode",
    "is_cell",
    "high_confidence",
    "full_length",
    "productive",
    "chain",
    "v_gene",
    "j_gene",
    "cdr3",
    "cdr3_nt",
    "umis",
}


@dataclass(frozen=True)
class PreparedCoreInputs:
    """Exact matrices and aligned metadata consumed by ``benisse_core.run_admm``."""

    expression: pd.DataFrame
    cell_clone_ids: np.ndarray
    node_ids: tuple[str, ...]
    annotation: pd.DataFrame
    embedding: np.ndarray
    master_dist_e: np.ndarray
    phi: np.ndarray
    si_matrix: np.ndarray
    identity: np.ndarray
    initial_a: np.ndarray
    ls_matrix: np.ndarray

    @property
    def n_cells(self) -> int:
        return int(self.expression.shape[1])

    @property
    def n_nodes(self) -> int:
        return len(self.node_ids)


def normalize_barcode(value: str) -> str:
    """Match the 10x barcode normalization used by R ``read.csv``/``prepare.R``."""
    return str(value).replace("-", ".")


def _require_columns(frame: pd.DataFrame, required, name: str) -> None:
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} missing required columns {sorted(missing)}")


def _true_like(series: pd.Series) -> pd.Series:
    return series.map(lambda value: str(value) == "True").astype(bool)


def _numeric_expression(expression: pd.DataFrame) -> pd.DataFrame:
    result = expression.copy()
    result.columns = [normalize_barcode(item) for item in result.columns]
    if len(set(result.columns)) != len(result.columns):
        raise ValueError("expression barcodes collide after '-' to '.' normalization")
    result = result.apply(pd.to_numeric, errors="raise")
    values = result.to_numpy(dtype="float64")
    if not np.isfinite(values).all():
        raise ValueError("expression contains non-finite values")
    return result


def _encoder_lookup(encoded: pd.DataFrame) -> pd.DataFrame:
    _require_columns(encoded, {"index", *ENCODER_COLUMNS}, "encoded input")
    if encoded["index"].isna().any():
        raise ValueError("encoded input contains a missing CDR3 index")
    if encoded["index"].astype(str).duplicated().any():
        raise ValueError("encoded input contains duplicate CDR3 indices")
    lookup = encoded.set_index(encoded["index"].astype(str))[list(ENCODER_COLUMNS)]
    lookup = lookup.apply(pd.to_numeric, errors="raise")
    if not np.isfinite(lookup.to_numpy(dtype="float64")).all():
        raise ValueError("encoded input contains non-finite features")
    return lookup


def clone_expression_distances(cleaned_exp, clonality_labels, node_ids) -> np.ndarray:
    """Port ``initiation.R``'s clone-level expression-distance calculation."""
    if isinstance(cleaned_exp, (str, Path)):
        expression = pd.read_csv(cleaned_exp, sep=r"\s+", index_col=0)
    else:
        expression = pd.DataFrame(cleaned_exp).copy()
    if isinstance(clonality_labels, (str, Path)):
        labels = pd.read_csv(clonality_labels, header=None).iloc[:, 0].astype(str).to_numpy()
    else:
        labels = np.asarray(clonality_labels, dtype=str)
    values = expression.to_numpy(dtype="float64")
    if values.ndim != 2 or values.shape[1] != labels.size:
        raise ValueError("expression columns and clonality labels must have equal length")
    if labels.size < 2 or not np.isfinite(values).all():
        raise ValueError("expression data require at least two finite cells")

    deviations = np.std(values, axis=1, ddof=1)
    keep = deviations > np.quantile(deviations, 0.9)
    if np.count_nonzero(keep) < 1:
        raise ValueError("no genes remain after variable-gene selection")
    cell_matrix = values[keep].T
    n_components = min(10, cell_matrix.shape[0], cell_matrix.shape[1])
    scores = PCA(n_components=n_components, svd_solver="full").fit_transform(cell_matrix)
    cell_distances = squareform(pdist(scores))

    unique_labels = np.asarray(sorted(set(labels)))
    indicator = sparse.csr_matrix(
        (
            np.ones(labels.size),
            (np.arange(labels.size), np.searchsorted(unique_labels, labels)),
        ),
        shape=(labels.size, unique_labels.size),
    )
    aggregated = np.asarray(indicator.T @ cell_distances @ indicator)
    scaler = 4 * aggregated.sum() / (labels.size * (labels.size - 1))
    if scaler <= 0 or not np.isfinite(scaler):
        raise ValueError("expression-distance scaling is zero or non-finite")
    clone_sizes = np.asarray(indicator.sum(axis=0)).ravel()
    normalized = aggregated / clone_sizes[:, None] / clone_sizes[None, :] / scaler
    positions = {name: i for i, name in enumerate(unique_labels)}
    missing = [name for name in node_ids if name not in positions]
    if missing:
        raise ValueError(f"clonality labels are missing {len(missing)} graph nodes")
    order = np.asarray([positions[name] for name in node_ids])
    result = normalized[np.ix_(order, order)]
    return (result + result.T) / 2


def initialize_core_inputs(
    expression: pd.DataFrame,
    cell_clone_ids: Sequence[str],
    annotation: pd.DataFrame,
    embedding,
    hyperparameters: HyperParameters,
) -> PreparedCoreInputs:
    """Build ``phi``, ``SI``, and ``LS`` from already aligned clone data."""
    expression = _numeric_expression(expression)
    cell_clone_ids = np.asarray(cell_clone_ids, dtype=str)
    annotation = annotation.copy()
    embedding = np.asarray(embedding, dtype="float64")
    _require_columns(annotation, {"v_gene", "j_gene", "cdr3"}, "annotation")
    if expression.shape[1] != cell_clone_ids.size:
        raise ValueError("expression columns and cell clone ids must align")
    if annotation.index.duplicated().any():
        raise ValueError("annotation node ids must be unique")
    node_ids = tuple(map(str, annotation.index))
    if embedding.shape != (len(node_ids), 20):
        raise ValueError(f"embedding shape {embedding.shape} != ({len(node_ids)}, 20)")
    if not np.isfinite(embedding).all():
        raise ValueError("embedding contains non-finite values")
    if set(cell_clone_ids) != set(node_ids):
        raise ValueError("cell clone ids and annotation node ids have different membership")

    master_dist_e = clone_expression_distances(expression, cell_clone_ids, node_ids)
    phi = squareform(pdist(embedding)) ** 2
    v_gene = annotation["v_gene"].astype("string").str.strip()
    j_gene = annotation["j_gene"].astype("string").str.strip()
    incomplete = v_gene.isna() | j_gene.isna() | v_gene.eq("") | j_gene.eq("")
    if incomplete.any():
        raise ValueError(
            "annotation contains missing V/J calls; incomplete calls cannot define "
            "Benisse candidate-edge support"
        )
    families = (v_gene + "\x1f" + j_gene).to_numpy(dtype=str)
    si_matrix = (families[:, None] == families[None, :]).astype("float64")
    np.fill_diagonal(si_matrix, 0)
    identity = np.eye(len(node_ids), dtype="float64")
    initial_a = hyperparameters.lambda1 * (1 - identity) * si_matrix
    weighted_expression = master_dist_e * si_matrix
    if si_matrix.sum() == 0:
        ls_matrix = np.zeros_like(si_matrix)
    else:
        ls_matrix = -(
            np.diag(weighted_expression.sum(axis=1)) - weighted_expression
        ) / si_matrix.sum()
    return PreparedCoreInputs(
        expression=expression,
        cell_clone_ids=cell_clone_ids,
        node_ids=node_ids,
        annotation=annotation,
        embedding=embedding,
        master_dist_e=master_dist_e,
        phi=phi,
        si_matrix=si_matrix,
        identity=identity,
        initial_a=initial_a,
        ls_matrix=ls_matrix,
    )


def prepare_frames(
    expression: pd.DataFrame,
    contigs: pd.DataFrame,
    encoded: pd.DataFrame,
    hyperparameters: HyperParameters,
    *,
    rm_cutoff: int | None = None,
) -> PreparedCoreInputs:
    """Port the standard-CSV path in ``Benisse.R`` without invoking R."""
    expression = _numeric_expression(expression)
    contigs = contigs.copy()
    _require_columns(contigs, REQUIRED_CONTIG_COLUMNS, "contigs")
    lookup = _encoder_lookup(encoded)
    contigs["barcode"] = contigs["barcode"].astype(str).map(normalize_barcode)
    contigs = contigs[contigs["cdr3"].astype(str).isin(lookup.index)].copy()
    quality = (
        _true_like(contigs["is_cell"])
        & _true_like(contigs["high_confidence"])
        & _true_like(contigs["full_length"])
        & _true_like(contigs["productive"])
        & (contigs["chain"].astype(str) == "IGH")
    )
    heavy = contigs[quality].copy()
    heavy = heavy[heavy["cdr3"].notna() & heavy["cdr3_nt"].notna()]
    heavy["umis"] = pd.to_numeric(heavy["umis"], errors="coerce")
    heavy = heavy.sort_values(
        ["barcode", "umis"], ascending=[False, False], kind="mergesort"
    ).drop_duplicates("barcode", keep="first")

    # R's intersect/match sequence restores expression-column order after the
    # descending sort used to choose the maximum-UMI contig.
    heavy = heavy.set_index("barcode", drop=False)
    selected_barcodes = [barcode for barcode in expression.columns if barcode in heavy.index]
    heavy = heavy.loc[selected_barcodes].copy()
    if heavy.empty:
        raise ValueError("no expression-aligned productive IGH contigs remain")
    expression = expression.loc[:, selected_barcodes]
    cell_clone_ids = (
        heavy["v_gene"].astype(str)
        + "_"
        + heavy["cdr3"].astype(str)
        + "_"
        + heavy["j_gene"].astype(str)
    ).to_numpy()
    first = ~pd.Series(cell_clone_ids).duplicated().to_numpy()
    node_ids = cell_clone_ids[first]
    representative = heavy.iloc[np.flatnonzero(first)].copy()
    annotation = representative[["v_gene", "j_gene", "cdr3"]].copy()
    annotation["barcode"] = representative.index.astype(str)
    annotation.index = pd.Index(node_ids, name="clone")
    embedding = lookup.loc[representative["cdr3"].astype(str)].to_numpy(dtype="float64")

    if rm_cutoff is not None:
        if not isinstance(rm_cutoff, int) or rm_cutoff < 1:
            raise ValueError("rm_cutoff must be a positive integer")
        families = annotation["v_gene"].astype(str) + "\x1f" + annotation["j_gene"].astype(str)
        keep_families = set(families.value_counts()[lambda values: values >= rm_cutoff].index)
        keep_nodes = families.isin(keep_families).to_numpy()
        kept_ids = set(annotation.index[keep_nodes])
        keep_cells = np.asarray([item in kept_ids for item in cell_clone_ids])
        expression = expression.loc[:, keep_cells]
        cell_clone_ids = cell_clone_ids[keep_cells]
        annotation = annotation.iloc[np.flatnonzero(keep_nodes)].copy()
        embedding = embedding[keep_nodes]
        if annotation.empty:
            raise ValueError("rm_cutoff removed every V/J candidate family")

    clone_sizes = pd.Series(cell_clone_ids).value_counts()
    annotation["clsize"] = [int(clone_sizes[item]) for item in annotation.index]
    return initialize_core_inputs(
        expression,
        cell_clone_ids,
        annotation,
        embedding,
        hyperparameters,
    )


def prepare_csv_inputs(
    expression_csv,
    contigs_csv,
    encoded_csv,
    hyperparameters: HyperParameters,
    *,
    rm_cutoff: int | None = None,
) -> PreparedCoreInputs:
    """Read standard Benisse CSVs and return R-free initialized core inputs."""
    expression = pd.read_csv(expression_csv, index_col=0)
    contigs = pd.read_csv(contigs_csv)
    encoded = pd.read_csv(encoded_csv)
    return prepare_frames(
        expression,
        contigs,
        encoded,
        hyperparameters,
        rm_cutoff=rm_cutoff,
    )
