"""Phase 4c Python->R bridge for the Benisse core (INTERNAL, PRE-API).

Exposes the Benisse R core from Python as an import-and-call interface, without
betting correctness on an unproven port: it runs the **unmodified** ``Benisse.R``
as a subprocess and parses its outputs into the Phase 4b
:class:`airr_adapter.BenisseNetworkResult` schema. This is the interim
scientific interface described in ``UPDATE_PLAN.md`` Phase 4c; the polished
public/scverse API and the Python core port (4d/4e) come later.

Design (see the 4c discussion and PHASE4C_NOTES.md):

* subprocess, not rpy2 -- rpy2 on this Intel-mac + R 4.3 stack is fragile and
  buys nothing the single-function-call wrapper does not already give. The R
  invocation is an implementation detail hidden behind :func:`run_benisse_r`.
* The encoder is called by import (``from AchillesEncoder import encode_bcr``),
  the scverse-idiomatic way, not via its CLI.
* Graph nodes are Benisse's deduplicated clone keys (``v_gene_cdr3_j_gene``),
  taken from the R side (``clone_annotation.csv``), NOT per-cell contigs -- see
  the node-identity correction in ``airr_adapter`` / PHASE4B_NOTES.md #6.

:func:`read_benisse_network` is deliberately standalone so the Phase 4d Python
port can validate against the same parsed-oracle representation.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from airr_adapter import BenisseNetworkResult, validate_network_result, benisse_provenance

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_BENISSE_R = REPO_ROOT / "Benisse.R"

#: Files ``Benisse.R`` writes into its output directory.
OUTPUT_FILES = {
    "clone_annotation": "clone_annotation.csv",
    "cleaned_exp": "cleaned_exp.txt",
    "clonality_label": "clonality_label.txt",
    "sparse_graph": "sparse_graph.txt",
    "latent_dist": "latent_dist.txt",
    "results_rdata": "Benisse_results.RData",
    "connection_plot": "connectionplot.pdf",
    "cross_dist_plot": "in_cross_dist_check.pdf",
}


@dataclass
class BenisseRParams:
    """Positional parameters for ``Benisse.R`` (CLI args 5-11).

    Defaults reproduce the committed ``example/`` run. ``lambda2`` scales with the
    dataset (the example uses 1610 ~= the BCR-cell count), so set it per dataset
    rather than trusting the default.
    """

    lambda2: float = 1610
    gamma: float = 1
    max_iter: int = 100
    lambda1: float = 1
    rho: float = 1
    m: int = 10
    stop_cutoff: float = 1e-10

    def as_cli_args(self) -> list[str]:
        # Order matches Benisse.R args[5..11].
        return [
            _fmt(self.lambda2), _fmt(self.gamma), _fmt(self.max_iter),
            _fmt(self.lambda1), _fmt(self.rho), _fmt(self.m), _fmt(self.stop_cutoff),
        ]


def _fmt(value: Any) -> str:
    """Format a numeric CLI arg without spurious trailing ``.0`` for integers."""
    if isinstance(value, bool):
        raise TypeError("boolean is not a valid Benisse.R numeric parameter")
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and value.is_integer() and abs(value) < 1e15:
        return str(int(value))
    return repr(value) if isinstance(value, float) else str(value)


# --- Running the unmodified Benisse.R ------------------------------------


def build_benisse_command(
    exp_csv, contigs_csv, encoded_csv, out_dir, params: BenisseRParams,
    *, benisse_r=DEFAULT_BENISSE_R, rscript: str = "Rscript",
) -> list[str]:
    """Assemble the ``Rscript Benisse.R ...`` argv (no side effects)."""
    return [
        rscript, str(benisse_r),
        str(exp_csv), str(contigs_csv), str(encoded_csv), str(out_dir),
        *params.as_cli_args(),
    ]


def run_benisse_r(
    exp_csv, contigs_csv, encoded_csv, out_dir, params: BenisseRParams | None = None,
    *, benisse_r=DEFAULT_BENISSE_R, rscript: str = "Rscript",
    timeout: float | None = None, check: bool = True,
) -> dict[str, Path]:
    """Run the unmodified ``Benisse.R`` on the three input CSVs.

    Returns a mapping of the output file paths under *out_dir* (see
    :data:`OUTPUT_FILES`). Raises ``FileNotFoundError`` for a missing input/script
    and ``RuntimeError`` (with captured stderr) on a non-zero R exit.
    """
    params = params or BenisseRParams()
    exp_csv, contigs_csv, encoded_csv = map(Path, (exp_csv, contigs_csv, encoded_csv))
    benisse_r = Path(benisse_r)
    for path in (exp_csv, contigs_csv, encoded_csv, benisse_r):
        if not path.exists():
            raise FileNotFoundError(f"required input does not exist: {path}")
    if shutil.which(rscript) is None:
        raise FileNotFoundError(
            f"Rscript executable not found: {rscript!r}. Install R or pass rscript=."
        )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = build_benisse_command(
        exp_csv, contigs_csv, encoded_csv, out_dir, params,
        benisse_r=benisse_r, rscript=rscript,
    )
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"Benisse.R failed (exit {proc.returncode}).\n"
            f"command: {' '.join(cmd)}\n"
            f"--- stderr (tail) ---\n{_tail(proc.stderr)}"
        )
    return {name: out_dir / fname for name, fname in OUTPUT_FILES.items()}


def _tail(text: str, lines: int = 40) -> str:
    return "\n".join((text or "").splitlines()[-lines:])


# --- Encoder (imported, not CLI) -----------------------------------------


def encode_bcr_csv(encoder_input_csv, encoded_csv, *, cuda: bool = False) -> Path:
    """Run the Benisse encoder on an encoder-input CSV via ``encode_bcr`` import.

    *encoder_input_csv* has the ``contigs``/``cdr3`` columns produced by
    :func:`airr_adapter.build_encoder_input`. Returns the encoded CSV path.
    """
    from AchillesEncoder import encode_bcr

    encoded_csv = Path(encoded_csv)
    encode_bcr(str(encoder_input_csv), str(encoded_csv), cuda=cuda)
    return encoded_csv


# --- Parsing R output into the 4b schema ---------------------------------


def read_clone_annotation(out_dir_or_file) -> pd.DataFrame:
    """Read ``clone_annotation.csv`` (node metadata, one row per graph node)."""
    path = _resolve(out_dir_or_file, "clone_annotation")
    ann = pd.read_csv(path, index_col=0)
    required = {"v_gene", "j_gene", "cdr3"}
    missing = required - set(ann.columns)
    if missing:
        raise ValueError(f"clone_annotation.csv missing columns {sorted(missing)}.")
    return ann


def clone_keys(annotation: pd.DataFrame) -> list[str]:
    """Benisse clone keys ``v_gene_cdr3_j_gene`` in graph-node order."""
    return [
        f"{v}_{c}_{j}"
        for v, c, j in zip(annotation["v_gene"], annotation["cdr3"], annotation["j_gene"])
    ]


def read_sparse_graph(out_dir_or_file) -> np.ndarray:
    """Read ``sparse_graph.txt`` (dense, space-separated ``results$A``)."""
    path = _resolve(out_dir_or_file, "sparse_graph")
    # np.loadtxt returns a 0-d scalar for a 1x1 matrix and a 1-d row for a single
    # line; atleast_2d normalises the 1-node case while a genuine 1xN row still
    # fails the square check below.
    A = np.atleast_2d(np.loadtxt(path))
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"sparse_graph.txt is not square: shape {A.shape}.")
    return A


def read_benisse_network(
    out_dir, *, params: BenisseRParams | None = None,
    provenance: dict[str, Any] | None = None, symmetry_atol: float = 1e-8,
) -> BenisseNetworkResult:
    """Parse ``Benisse.R`` outputs into a validated ``BenisseNetworkResult``.

    Nodes are the deduplicated clone keys from ``clone_annotation.csv``; edges are
    the upper triangle of the symmetric adjacency ``A`` (``sparse_graph.txt``),
    matching the schema's undirected convention. Asserts ``A`` is square,
    symmetric, and dimension-aligned to the annotation.
    """
    A = read_sparse_graph(out_dir)
    annotation = read_clone_annotation(out_dir)
    if len(annotation) != A.shape[0]:
        raise ValueError(
            f"clone_annotation rows ({len(annotation)}) != adjacency dim ({A.shape[0]})."
        )
    if not np.isfinite(A).all():
        raise ValueError("adjacency A contains non-finite (NaN/inf) values.")
    if not np.allclose(A, A.T, atol=symmetry_atol):
        raise ValueError("adjacency A is not symmetric; expected an undirected graph.")
    if not np.allclose(np.diag(A), 0.0, atol=symmetry_atol):
        raise ValueError("adjacency A has a nonzero diagonal (self-loops); expected zero.")

    nodes = clone_keys(annotation)
    upper = np.triu(A, k=1)  # k=1 drops the (zero) diagonal / self-loops
    row, col = np.nonzero(upper)
    weight = upper[row, col]

    prov = benisse_provenance(
        source_dir=str(Path(out_dir)),
        n_nodes=int(A.shape[0]),
        n_edges=int(row.size),
        weight_min=float(weight.min()) if weight.size else None,
        weight_max=float(weight.max()) if weight.size else None,
        **(provenance or {}),
    )
    result = BenisseNetworkResult(
        node_ids=nodes,
        row=row.astype("int64"),
        col=col.astype("int64"),
        weight=weight.astype("float64"),
        params=asdict(params) if params else {},
        provenance=prov,
        directed=False,
    )
    validate_network_result(result)
    return result


# --- End-to-end (from standard Benisse CSV inputs) -----------------------


def run_pipeline(
    encoder_input_csv, exp_csv, contigs_csv, out_dir,
    params: BenisseRParams | None = None, *, cuda: bool = False,
    benisse_r=DEFAULT_BENISSE_R, rscript: str = "Rscript",
    timeout: float | None = None,
) -> BenisseNetworkResult:
    """Encode CDR3H, run the R core, and parse the clonotype network.

    End-to-end Python UX around the unmodified core: encodes *encoder_input_csv*
    (``contigs``/``cdr3``) with the Benisse encoder, runs ``Benisse.R`` against the
    expression and contigs CSVs, and returns the parsed
    :class:`~airr_adapter.BenisseNetworkResult`.
    """
    params = params or BenisseRParams()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    encoded_csv = encode_bcr_csv(encoder_input_csv, out_dir / "encoded.csv", cuda=cuda)
    run_benisse_r(
        exp_csv, contigs_csv, encoded_csv, out_dir, params,
        benisse_r=benisse_r, rscript=rscript, timeout=timeout,
    )
    return read_benisse_network(out_dir, params=params)


def _resolve(out_dir_or_file, key: str) -> Path:
    """Accept either a directory (join the known filename) or a direct file path."""
    path = Path(out_dir_or_file)
    if path.is_dir():
        return path / OUTPUT_FILES[key]
    return path
