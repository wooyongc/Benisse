"""Optional BCR embedding backends for the corrected Python pipeline.

The published Achilles encoder remains the default and is delegated to without
changing its behavior.  The experimental ESM-2 backend consumes the same
``contigs,cdr3`` input table, where ``cdr3`` is the heavy-chain amino-acid
sequence, and writes the same lookup-oriented CSV shape with a variable number
of feature columns.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from importlib import metadata
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd


DEFAULT_ESM2_MODEL = "esm2_t12_35M_UR50D"
_CANONICAL_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")


@dataclass(frozen=True)
class ESM2ModelSpec:
    loader: str
    final_layer: int
    dimension: int


ESM2_MODELS = {
    "esm2_t6_8M_UR50D": ESM2ModelSpec("esm2_t6_8M_UR50D", 6, 320),
    DEFAULT_ESM2_MODEL: ESM2ModelSpec(DEFAULT_ESM2_MODEL, 12, 480),
    "esm2_t30_150M_UR50D": ESM2ModelSpec("esm2_t30_150M_UR50D", 30, 640),
    "esm2_t33_650M_UR50D": ESM2ModelSpec("esm2_t33_650M_UR50D", 33, 1280),
}


def _strtobool(value: str) -> bool:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    raise argparse.ArgumentTypeError("expected True or False")


def _input_csvs(input_data) -> list[Path]:
    path = Path(input_data)
    if path.is_file():
        if path.suffix.lower() != ".csv":
            raise ValueError(f"embedding input must be a CSV file: {path}")
        return [path]
    if path.is_dir():
        files = sorted(item for item in path.rglob("*.csv") if item.is_file())
        if files:
            return files
        raise ValueError(f"embedding input directory contains no CSV files: {path}")
    raise ValueError(f"embedding input does not exist: {path}")


def _read_esm2_sequences(input_data) -> list[str]:
    rows = []
    for path in _input_csvs(input_data):
        frame = pd.read_csv(path)
        missing = {"contigs", "cdr3"} - set(frame.columns)
        if missing:
            raise ValueError(f"{path} missing required columns {sorted(missing)}")
        selected = frame[["contigs", "cdr3"]].copy()
        selected["source_file"] = str(path)
        selected["source_row"] = np.arange(2, len(selected) + 2)
        rows.append(selected)
    combined = pd.concat(rows, ignore_index=True)
    if combined.empty:
        raise ValueError("embedding input contains no contigs/CDR3 sequences")
    missing_mask = combined[["contigs", "cdr3"]].isna()
    if missing_mask.any(axis=None):
        row_index = int(np.flatnonzero(missing_mask.any(axis=1).to_numpy())[0])
        item = combined.iloc[row_index]
        missing_columns = missing_mask.columns[missing_mask.iloc[row_index]].tolist()
        raise ValueError(
            f"missing required value(s) {missing_columns} at "
            f"{item['source_file']}:{item['source_row']}"
        )
    combined["cdr3"] = combined["cdr3"].astype(str).str.strip().str.upper()

    for item in combined.itertuples(index=False):
        sequence = item.cdr3
        location = f"{item.source_file}:{item.source_row}"
        if not sequence:
            raise ValueError(f"empty CDR3 amino-acid sequence at {location}")
        invalid = sorted(set(sequence) - _CANONICAL_AMINO_ACIDS)
        if invalid:
            raise ValueError(
                f"noncanonical CDR3 amino-acid sequence at {location}: "
                f"unexpected characters {invalid}"
            )
        if len(sequence) > 1022:
            raise ValueError(
                f"CDR3 amino-acid sequence at {location} has length {len(sequence)}; "
                "ESM-2 accepts at most 1022 residues"
            )
    return sorted(combined["cdr3"].drop_duplicates().tolist())


def _optional_esm_modules():
    try:
        version = metadata.version("fair-esm")
        import esm
        import torch
    except (metadata.PackageNotFoundError, ImportError) as exc:
        raise RuntimeError(
            "The ESM-2 backend requires the optional dependency "
            "`fair-esm==2.0.0`; install requirements-esm2.txt."
        ) from exc
    if version != "2.0.0" or not hasattr(esm, "pretrained"):
        raise RuntimeError(
            "The ESM-2 backend is tested with `fair-esm==2.0.0`; "
            f"found {version!r} or an incompatible `esm` module."
        )
    return esm, torch, version


def _cache_paths(torch_module, model_name: str) -> tuple[Path, Path]:
    checkpoint_dir = Path(torch_module.hub.get_dir()) / "checkpoints"
    return (
        checkpoint_dir / f"{model_name}.pt",
        checkpoint_dir / f"{model_name}-contact-regression.pt",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def esm2_metadata_path(output_data) -> Path:
    """Return the provenance sidecar path for an ESM-2 embedding CSV."""
    return Path(f"{Path(output_data)}.metadata.json")


def _encode_esm2(
    input_data,
    output_data,
    *,
    model_name: str,
    cuda: bool,
    batch_size: int,
    offline: bool,
) -> pd.DataFrame:
    if model_name not in ESM2_MODELS:
        raise ValueError(
            f"unsupported ESM-2 model {model_name!r}; "
            f"choose one of {sorted(ESM2_MODELS)}"
        )
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")

    sequences = _read_esm2_sequences(input_data)
    esm, torch, fair_esm_version = _optional_esm_modules()
    spec = ESM2_MODELS[model_name]
    model_path, regression_path = _cache_paths(torch, model_name)
    if offline:
        absent = [str(path) for path in (model_path, regression_path) if not path.is_file()]
        if absent:
            raise RuntimeError(
                "offline ESM-2 loading requires cached checkpoint files: "
                + ", ".join(absent)
            )

    loader = getattr(esm.pretrained, spec.loader)
    model, alphabet = loader()
    if not model_path.is_file() or not regression_path.is_file():
        raise RuntimeError("fair-esm loaded without leaving reproducible cached checkpoints")
    if getattr(model, "num_layers", None) != spec.final_layer:
        raise RuntimeError(
            f"{model_name} reported {getattr(model, 'num_layers', None)} layers; "
            f"expected {spec.final_layer}"
        )
    if getattr(model, "embed_dim", None) != spec.dimension:
        raise RuntimeError(
            f"{model_name} reported dimension {getattr(model, 'embed_dim', None)}; "
            f"expected {spec.dimension}"
        )

    if cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("cuda=True was requested, but CUDA is unavailable")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = model.to(device)
    model.eval()
    batch_converter = alphabet.get_batch_converter()

    vectors = []
    with torch.inference_mode():
        for start in range(0, len(sequences), batch_size):
            batch_sequences = sequences[start : start + batch_size]
            _, _, tokens = batch_converter(
                [(str(start + offset), sequence) for offset, sequence in enumerate(batch_sequences)]
            )
            tokens = tokens.to(device)
            output = model(
                tokens,
                repr_layers=[spec.final_layer],
                return_contacts=False,
            )
            representations = output["representations"][spec.final_layer]
            for row, sequence in enumerate(batch_sequences):
                vector = (
                    representations[row, 1 : len(sequence) + 1]
                    .to(dtype=torch.float64)
                    .mean(dim=0)
                    .cpu()
                    .numpy()
                )
                norm = float(np.linalg.norm(vector))
                if not np.isfinite(vector).all() or not np.isfinite(norm):
                    raise RuntimeError(f"{model_name} produced a non-finite embedding")
                if norm <= np.finfo(np.float64).eps:
                    raise RuntimeError(f"{model_name} produced a zero-norm embedding")
                vectors.append(vector / norm)

    matrix = np.asarray(vectors, dtype="float64")
    if matrix.shape != (len(sequences), spec.dimension):
        raise RuntimeError(
            f"{model_name} embedding shape {matrix.shape} != "
            f"({len(sequences)}, {spec.dimension})"
        )
    encoded = pd.DataFrame(matrix, columns=range(spec.dimension))
    encoded["index"] = sequences
    output_path = Path(output_data)
    encoded.to_csv(output_path)

    cache_root = Path(torch.hub.get_dir())
    sidecar = {
        "backend": "esm2",
        "model": model_name,
        "fair_esm_version": fair_esm_version,
        "final_layer": spec.final_layer,
        "embedding_dimension": spec.dimension,
        "pooling": "mean_residue",
        "normalization": "l2",
        "dtype": "float64",
        "checkpoints": {
            "model": {
                "path": str(model_path.relative_to(cache_root)),
                "sha256": _sha256(model_path),
            },
            "contact_regression": {
                "path": str(regression_path.relative_to(cache_root)),
                "sha256": _sha256(regression_path),
            },
        },
    }
    esm2_metadata_path(output_path).write_text(json.dumps(sidecar, indent=2) + "\n")
    return encoded


def encode_bcr_embeddings(
    input_data,
    output_data,
    *,
    backend: str = "achilles",
    model: str | None = None,
    cuda: bool = False,
    batch_size: int = 64,
    offline: bool = False,
) -> pd.DataFrame:
    """Encode BCR CDR3H amino-acid sequences with the selected backend."""
    if backend == "achilles":
        if model is not None:
            raise ValueError("model is only valid for backend='esm2'")
        if offline:
            raise ValueError("offline is only valid for backend='esm2'")
        from AchillesEncoder import encode_bcr

        return encode_bcr(input_data, output_data, cuda=cuda)
    if backend == "esm2":
        return _encode_esm2(
            input_data,
            output_data,
            model_name=model or DEFAULT_ESM2_MODEL,
            cuda=cuda,
            batch_size=batch_size,
            offline=offline,
        )
    raise ValueError("backend must be 'achilles' or 'esm2'")


def parse_option(argv=None):
    parser = argparse.ArgumentParser("Generate BCR CDR3H embeddings")
    parser.add_argument("--input_data", required=True)
    parser.add_argument("--output_data", required=True)
    parser.add_argument("--backend", choices=("achilles", "esm2"), default="achilles")
    parser.add_argument("--model", choices=tuple(ESM2_MODELS), default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--cuda", type=_strtobool, default=False)
    parser.add_argument("--offline", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    options = parse_option(argv)
    encode_bcr_embeddings(
        options.input_data,
        options.output_data,
        backend=options.backend,
        model=options.model,
        cuda=options.cuda,
        batch_size=options.batch_size,
        offline=options.offline,
    )


if __name__ == "__main__":
    main()
