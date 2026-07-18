"""Phase 4b AIRR/scverse contract adapter for Benisse (INTERNAL, PRE-API).

This module is the boundary between an AIRR/scverse object (a scirpy-style
``AnnData`` with ``obsm["airr"]``, or a ``MuData`` with an ``airr`` modality) and
the legacy Benisse encoder input contract. It is deliberately *not* a public
package or a stable API: it exists so Phase 4b behaviour (deterministic
heavy-chain selection, barcode joining, field mapping, MuData/AnnData
preservation, and the clonotype-network result schema) can be defined and tested
independently of Phase 4a, and later promoted into the package during Phase 4e.

Scope decisions (see ``airr_context.md`` and ``UPDATE_PLAN.md`` Phase 4b):

* AIRR Standards 2.0 field names at the receptor-data boundary.
* Scirpy's in-memory representation is the source object; we never invent a
  second AIRR container.
* Cell barcodes (AIRR ``cell_id``) stay canonical. Any normalisation needed to
  join to an R-style expression table (``-`` -> ``.``) is explicit and
  reversible, never an in-place mutation of the object.
* The encoder consumes only two columns, ``contigs`` (a unique per-chain id) and
  ``cdr3`` (the heavy-chain junction amino-acid sequence). See
  ``CMC/data_pre.py:load_BCRdata2`` which reads ``f[['contigs','cdr3']]``.

Runtime deps: numpy, pandas, awkward, scipy. AnnData/MuData are only needed by
callers that pass those objects; this module accepts either the container or its
raw ``obsm['airr']`` awkward array so it can be unit-tested cheaply.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import awkward as ak
import numpy as np
import pandas as pd

SCHEMA_VERSION = "benisse-airr-0.1"

# --- Field mappings -------------------------------------------------------

#: 10x Cell Ranger contig column -> AIRR 2.0 field. Matches scirpy's
#: ``read_10x_vdj`` mapping (airr_context.md). Kept for documentation and for a
#: future raw-10x ingress path; the scverse objects we read are already AIRR.
TEN_X_TO_AIRR: dict[str, str] = {
    "barcode": "cell_id",
    "contig_id": "sequence_id",
    "chain": "locus",
    "cdr3_nt": "junction",
    "cdr3": "junction_aa",
    "v_gene": "v_call",
    "d_gene": "d_call",
    "j_gene": "j_call",
    "c_gene": "c_call",
    "umis": "umi_count",
    "reads": "consensus_count",
    "productive": "productive",
}

#: AIRR field -> encoder input column. The encoder only reads these two.
AIRR_TO_ENCODER: dict[str, str] = {
    "sequence_id": "contigs",
    "junction_aa": "cdr3",
}

#: Loci treated as the "heavy" / VDJ chain for BCR data. Benisse embeds the
#: heavy-chain CDR3H only. (TCR beta/delta would extend this; recorded as an
#: open assumption rather than silently supported.)
HEAVY_LOCI: tuple[str, ...] = ("IGH",)

#: Count fields used as chain-abundance evidence for ranking, best first.
#: The Stephenson AP4 fixture stores ``duplicate_count``/``consensus_count`` and
#: has no ``umi_count`` column, so we fall back across all three.
COUNT_PREFERENCE: tuple[str, ...] = ("umi_count", "duplicate_count", "consensus_count")

#: AIRR fields we surface from the ragged array (projected before flattening so
#: awkward's unnamed index field and large alignment strings are skipped).
_CHAIN_FIELDS: tuple[str, ...] = (
    "locus",
    "productive",
    "junction_aa",
    "junction",
    "sequence_id",
    "v_call",
    "d_call",
    "j_call",
    "c_call",
    "cdr3_aa",
    "umi_count",
    "duplicate_count",
    "consensus_count",
)


# --- Awkward AIRR access --------------------------------------------------


def _airr_array(obj: Any, modality: str = "airr") -> ak.Array:
    """Return the ragged per-cell chain awkward array from a container/array.

    Accepts a raw awkward array, an AnnData (uses ``obsm['airr']``), or a MuData
    (uses ``mod[modality].obsm['airr']``). Duck-typed to avoid importing
    anndata/mudata here.
    """
    if isinstance(obj, ak.Array):
        return obj
    # MuData: has ``.mod`` mapping of modalities.
    mod = getattr(obj, "mod", None)
    if isinstance(mod, Mapping) and modality in mod:
        return mod[modality].obsm["airr"]
    # AnnData: has ``.obsm``.
    obsm = getattr(obj, "obsm", None)
    if obsm is not None and "airr" in obsm:
        return obsm["airr"]
    raise TypeError(
        "Expected an awkward array, an AnnData with obsm['airr'], or a MuData "
        f"with a '{modality}' modality; got {type(obj)!r}."
    )


def _obs_names(obj: Any, modality: str = "airr") -> list[str]:
    """Cell barcodes (AIRR cell_id) aligned to the airr array rows."""
    if isinstance(obj, ak.Array):
        return [str(i) for i in range(len(obj))]
    mod = getattr(obj, "mod", None)
    if isinstance(mod, Mapping) and modality in mod:
        return list(map(str, mod[modality].obs_names))
    return list(map(str, obj.obs_names))


def airr_to_dataframe(obj: Any, modality: str = "airr") -> pd.DataFrame:
    """Flatten the ragged AIRR array into one row per chain.

    Columns: ``cell_id``, ``chain_pos`` (0-based position within the cell), and
    every field in :data:`_CHAIN_FIELDS` that the object actually carries. The
    result is deterministic: cell order follows ``obs_names`` and chain order
    follows storage order within each cell.

    Only the known fields are surfaced here; custom/AIRR-extension fields are not
    flattened, but they are not lost -- the source object is never mutated, so
    they remain available in ``obsm['airr']`` for export.
    """
    airr = _airr_array(obj, modality)
    cell_ids = _obs_names(obj, modality)
    present = [f for f in _CHAIN_FIELDS if f in airr.fields]
    projected = airr[present] if present else airr
    records = ak.to_list(projected)

    rows: list[dict[str, Any]] = []
    for cell_id, chains in zip(cell_ids, records):
        for pos, chain in enumerate(chains):
            row: dict[str, Any] = {"cell_id": cell_id, "chain_pos": pos}
            for f in present:
                row[f] = chain.get(f)
            rows.append(row)
    columns = ["cell_id", "chain_pos", *present]
    return pd.DataFrame(rows, columns=columns)


# --- Deterministic heavy-chain selection ----------------------------------


def _count_series(frame: pd.DataFrame) -> pd.Series:
    """Chain-abundance evidence, coalescing the preferred count fields."""
    result = pd.Series(0.0, index=frame.index)
    have_value = pd.Series(False, index=frame.index)
    for name in COUNT_PREFERENCE:
        if name not in frame.columns:
            continue
        col = pd.to_numeric(frame[name], errors="coerce")
        take = (~have_value) & col.notna()
        result = result.mask(take, col)
        have_value = have_value | col.notna()
    return result.fillna(0.0)


#: Values counted as AIRR ``productive`` true. AIRR TSV encodes booleans as
#: ``T``/``F`` strings, so a plain ``bool()`` cast is wrong (``bool("F")`` is
#: True); match explicitly instead (airr_context.md: "parse booleans, not string
#: equality").
_TRUTHY = {True, 1, "T", "TRUE", "True", "true", "t"}


def _parse_productive(series: pd.Series) -> pd.Series:
    """Boolean mask of truthy AIRR ``productive`` values (null -> False)."""
    return series.map(lambda v: v in _TRUTHY).astype(bool)


def _require_chain_fields(frame: pd.DataFrame, fields: Sequence[str]) -> None:
    missing = [f for f in fields if f not in frame.columns]
    if missing:
        raise ValueError(
            f"AIRR object is missing required chain field(s) {missing}; "
            f"available fields: {sorted(frame.columns)}."
        )


def select_heavy_chains(
    obj: Any,
    modality: str = "airr",
    heavy_loci: Sequence[str] = HEAVY_LOCI,
    require_productive: bool = True,
) -> pd.DataFrame:
    """Pick exactly one heavy chain per cell, deterministically.

    A candidate chain must have ``locus`` in *heavy_loci*, a non-empty
    ``junction_aa``, and (when *require_productive*) a truthy ``productive``.
    Candidates are ranked per cell by descending abundance count, then ascending
    ``junction_aa``, then ascending ``sequence_id``, then storage position. Given
    AIRR's guarantee that ``sequence_id`` is unique, this is a total order on
    chain content, so the winner is independent of storage order; the final
    position key only breaks ties between otherwise-identical records (which
    AIRR should not produce) and keeps the result deterministic for a given
    object. Cells with no candidate are dropped.

    Returns a DataFrame indexed by ``cell_id`` (unique) with the selected
    chain's fields plus a ``selection_count`` column recording the abundance
    value used. Row order follows the object's cell order.
    """
    chains = airr_to_dataframe(obj, modality)
    if chains.empty:
        return chains.set_index("cell_id") if "cell_id" in chains else chains

    required = ["locus", "junction_aa", "sequence_id"]
    if require_productive:
        required.append("productive")
    _require_chain_fields(chains, required)

    cand = chains[chains["locus"].isin(list(heavy_loci))].copy()
    ja = cand["junction_aa"].astype("string")
    cand = cand[ja.notna() & (ja.str.len() > 0)]
    if require_productive:
        cand = cand[_parse_productive(cand["productive"])]

    if cand.empty:
        return pd.DataFrame(columns=chains.columns.drop("cell_id")).rename_axis("cell_id")

    cand["selection_count"] = _count_series(cand)
    # Preserve original cell order for a stable output row order.
    cell_order = {cid: i for i, cid in enumerate(dict.fromkeys(chains["cell_id"]))}
    cand["_cell_order"] = cand["cell_id"].map(cell_order)
    cand["_ja"] = cand["junction_aa"].astype(str)
    cand["_sid"] = cand["sequence_id"].astype(str)

    cand = cand.sort_values(
        by=["_cell_order", "selection_count", "_ja", "_sid", "chain_pos"],
        ascending=[True, False, True, True, True],
        kind="mergesort",
    )
    winners = cand.drop_duplicates("cell_id", keep="first")
    winners = winners.drop(columns=["_cell_order", "_ja", "_sid", "chain_pos"])
    return winners.set_index("cell_id")


def scirpy_primary_heavy_index(obj: Any, modality: str = "airr") -> "pd.Series | None":
    """Scirpy's own primary heavy-chain pointer, if the object carries it.

    Returns a Series (index = cell_id) of the ``VDJ[0]`` chain position from
    ``obsm['chain_indices']``, or ``None`` when the object has no such array
    (e.g. a hand-built synthetic object). Used to cross-check that our
    independent ranking agrees with the ecosystem's deterministic selection.
    """
    if isinstance(obj, ak.Array):
        return None
    mod = getattr(obj, "mod", None)
    if isinstance(mod, Mapping) and modality in mod:
        adata = mod[modality]
    else:
        adata = obj
    obsm = getattr(adata, "obsm", None)
    if obsm is None or "chain_indices" not in obsm:
        return None
    ci = obsm["chain_indices"]
    vdj0 = ak.to_list(ci["VDJ"][:, 0])
    cell_ids = _obs_names(obj, modality)
    return pd.Series(vdj0, index=pd.Index(cell_ids, name="cell_id"), name="vdj0")


# --- Barcode joining ------------------------------------------------------


def to_r_barcode(barcode: str) -> str:
    """Normalise a canonical AIRR cell_id to the R-expression-table style.

    ``R/prepare.R`` replaces ``-`` with ``.`` to match its expression columns.
    This is reversible only when the barcode contains no ``.`` already; use
    :func:`assert_reversible_barcodes` to guard a batch before relying on it.
    """
    return barcode.replace("-", ".")


def from_r_barcode(barcode: str) -> str:
    """Inverse of :func:`to_r_barcode` (``.`` -> ``-``)."""
    return barcode.replace(".", "-")


def assert_reversible_barcodes(barcodes: Iterable[str]) -> None:
    """Raise if ``-``<->``.`` normalisation would not round-trip or would collide.

    Guards two failure modes: a pre-existing ``.`` (which breaks the inverse)
    and normalised collisions (two distinct barcodes mapping to one).
    """
    barcodes = list(barcodes)
    with_dot = [b for b in barcodes if "." in b]
    if with_dot:
        raise ValueError(
            f"{len(with_dot)} barcode(s) already contain '.', so '-'<->'.' "
            f"normalisation is not reversible (e.g. {with_dot[0]!r})."
        )
    normalised = [to_r_barcode(b) for b in barcodes]
    if len(set(normalised)) != len(set(barcodes)):
        raise ValueError("R-style barcode normalisation is not injective on this set.")
    for original in barcodes:
        if from_r_barcode(to_r_barcode(original)) != original:
            raise ValueError(f"Barcode {original!r} does not round-trip through R style.")


# --- Encoder input --------------------------------------------------------


def build_encoder_input(heavy: pd.DataFrame) -> pd.DataFrame:
    """Turn selected heavy chains into the encoder's input table.

    The returned frame is indexed by the canonical cell barcode and has exactly
    the two columns the encoder reads: ``contigs`` (<- ``sequence_id``) and
    ``cdr3`` (<- ``junction_aa``). Written with ``index=True`` this reproduces
    the committed example layout (``,contigs,cdr3``).
    """
    if heavy.index.name != "cell_id":
        raise ValueError("Expected select_heavy_chains() output indexed by cell_id.")
    out = pd.DataFrame(
        {
            "contigs": heavy["sequence_id"].astype(str).to_numpy(),
            "cdr3": heavy["junction_aa"].astype(str).to_numpy(),
        },
        index=heavy.index.rename("barcode"),
    )
    return out


def write_encoder_input_csv(heavy: pd.DataFrame, path) -> pd.DataFrame:
    """Write :func:`build_encoder_input` output to ``path`` and return it."""
    table = build_encoder_input(heavy)
    table.to_csv(path)
    return table


# --- Embedding preservation ----------------------------------------------


def attach_embedding(
    obj: Any,
    embedding: pd.DataFrame,
    modality: str = "airr",
    key: str = "X_benisse",
) -> Any:
    """Write a per-cell embedding into the AIRR modality without harming it.

    *embedding* is indexed by cell_id with one column per latent dimension. Rows
    are aligned to the modality's ``obs_names``; cells absent from *embedding*
    (e.g. no heavy chain) get an all-NaN row so the array stays cell-aligned.

    Coverage is checked so a barcode-namespace mismatch does not pass silently:
    if *no* embedding row matches an ``obs_name`` (the classic ``-``/``.``
    normalisation bug) a ``ValueError`` is raised; a partial mismatch warns. The
    reserved keys ``airr``/``chain_indices`` cannot be overwritten. Mutates and
    returns *obj*.
    """
    if key in ("airr", "chain_indices"):
        raise ValueError(f"Refusing to overwrite reserved AIRR obsm key {key!r}.")

    mod = getattr(obj, "mod", None)
    adata = mod[modality] if isinstance(mod, Mapping) and modality in mod else obj
    cell_ids = list(map(str, adata.obs_names))

    before_fields = list(adata.obsm["airr"].fields)
    before_len = len(adata.obsm["airr"])

    matched = embedding.index.isin(cell_ids)
    if len(embedding) and not matched.any():
        raise ValueError(
            "No embedding row matched obs_names; check the barcode namespace "
            "(e.g. R-style '.' vs canonical '-'). Nothing was written."
        )
    if not matched.all():
        warnings.warn(
            f"{(~matched).sum()} of {len(embedding)} embedding rows did not match "
            "any obs_name and were dropped.",
            stacklevel=2,
        )

    aligned = embedding.reindex(cell_ids)
    matrix = aligned.to_numpy(dtype="float64")
    if matrix.shape[0] != len(cell_ids):
        raise AssertionError("Embedding row count does not match obs_names.")
    adata.obsm[key] = matrix

    # obsm['airr'] is never written here; these guards catch accidental aliasing.
    assert list(adata.obsm["airr"].fields) == before_fields, "AIRR fields mutated"
    assert len(adata.obsm["airr"]) == before_len, "AIRR row count mutated"
    if "chain_indices" in adata.obsm:
        assert len(adata.obsm["chain_indices"]) == before_len, "chain_indices mutated"
    return obj


# --- Clonotype-network result schema --------------------------------------


@dataclass
class BenisseNetworkResult:
    """Benisse's scientific output: a sparse graph over receptor nodes.

    Benisse's object is a clonotype/receptor network, *not* a cell x cell
    matrix, so it is stored with an explicit ``node_ids`` index rather than in a
    cell-aligned ``obsp`` slot (airr_context.md). Edges are an explicit COO
    triple (``row``, ``col``, ``weight``) indexing into ``node_ids``.

    ``node_ids`` are Benisse's graph nodes, which are **deduplicated clones**,
    not cells or contigs: ``Benisse.R`` keys clones by ``v_gene _ cdr3 _ j_gene``,
    deduplicates, sorts them, and may drop clones below ``rm_cutoff`` before
    building ``results$A`` (``Benisse.R:64-73``, ``R/initiation.R``). So a
    consumer (the Phase 4c bridge) must take ``node_ids`` and their order from
    the R side (``clone_annotation.csv`` / ``meta_dedup``), NOT from the
    per-cell selection -- the two are 1:1 only when there is no clonal expansion
    and no cutoff (as happens to hold for the AP4 fixture).

    Benisse's graph is undirected. With ``directed=False`` (default), edges must
    be stored in canonical upper-triangle form (``row <= col``) so a symmetric
    R ``A`` is not double-counted; the 4c bridge upper-triangularises before
    constructing a result. ``params`` and ``provenance`` carry the R/model
    parameters, input hashes, and schema identity needed to reproduce the run.
    """

    node_ids: list[str]
    row: np.ndarray
    col: np.ndarray
    weight: np.ndarray
    params: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    directed: bool = False
    schema_version: str = SCHEMA_VERSION

    @property
    def n_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def n_edges(self) -> int:
        return int(len(self.row))

    def to_coo(self):
        from scipy.sparse import coo_matrix

        n = self.n_nodes
        return coo_matrix(
            (np.asarray(self.weight, dtype="float64"),
             (np.asarray(self.row, dtype="int64"), np.asarray(self.col, dtype="int64"))),
            shape=(n, n),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "node_ids": list(self.node_ids),
            "directed": bool(self.directed),
            "edges": {
                "row": np.asarray(self.row, dtype="int64"),
                "col": np.asarray(self.col, dtype="int64"),
                "weight": np.asarray(self.weight, dtype="float64"),
            },
            "params": self.params,
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BenisseNetworkResult":
        edges = data["edges"]
        return cls(
            node_ids=list(data["node_ids"]),
            row=np.asarray(edges["row"], dtype="int64"),
            col=np.asarray(edges["col"], dtype="int64"),
            weight=np.asarray(edges["weight"], dtype="float64"),
            params=dict(data.get("params", {})),
            provenance=dict(data.get("provenance", {})),
            directed=bool(data.get("directed", False)),
            schema_version=data.get("schema_version", SCHEMA_VERSION),
        )


def validate_network_result(result: BenisseNetworkResult) -> None:
    """Raise ``ValueError`` if *result* is not a well-formed network."""
    n = result.n_nodes
    if len(set(result.node_ids)) != n:
        raise ValueError("node_ids must be unique.")
    lengths = {len(result.row), len(result.col), len(result.weight)}
    if len(lengths) != 1:
        raise ValueError("row, col, and weight must have equal length.")

    row = np.asarray(result.row)
    col = np.asarray(result.col)
    weight = np.asarray(result.weight)
    if result.n_edges:
        for name, arr in (("row", row), ("col", col)):
            if arr.dtype.kind not in ("i", "u"):
                raise ValueError(
                    f"{name} must have an integer dtype (got {arr.dtype}); "
                    "float indices would be truncated by to_coo()."
                )
            if arr.min() < 0 or arr.max() >= n:
                raise ValueError(f"{name} index out of range for {n} nodes.")
        if not np.isfinite(weight).all():
            raise ValueError("weight contains NaN or infinite values.")
        if not result.directed and np.any(row > col):
            raise ValueError(
                "undirected result must be stored upper-triangular (row <= col); "
                "symmetrise the adjacency before constructing the result, or set "
                "directed=True."
            )
    if result.schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"Unexpected schema_version {result.schema_version!r}; "
            f"expected {SCHEMA_VERSION!r}."
        )


def attach_network_result(
    obj: Any, result: BenisseNetworkResult, uns_key: str = "benisse"
) -> Any:
    """Store *result* under ``uns[uns_key]['clonotype_network']``.

    Deliberately uses ``uns`` (a free-form slot) with an explicit node index,
    never a cell-aligned ``obsp``, because the network's nodes are receptors,
    not the object's cells. Mutates and returns *obj*.
    """
    validate_network_result(result)
    block = obj.uns.get(uns_key, {})
    if not isinstance(block, dict):
        block = {}
    block["clonotype_network"] = result.to_dict()
    block["schema_version"] = result.schema_version
    obj.uns[uns_key] = block
    return obj


def read_network_result(
    obj: Any, uns_key: str = "benisse"
) -> BenisseNetworkResult:
    """Inverse of :func:`attach_network_result`."""
    block = obj.uns[uns_key]["clonotype_network"]
    return BenisseNetworkResult.from_dict(block)


def benisse_provenance(**items: Any) -> dict[str, Any]:
    """Build a JSON-serialisable provenance dict, rejecting non-serialisable values."""
    json.dumps(items)  # fail fast if a caller passes something unserialisable
    return dict(items)
