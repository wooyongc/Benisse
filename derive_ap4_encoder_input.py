"""Reproducibly derive the Benisse encoder input from the AP4 AIRR fixture.

Phase 4b groundwork script. Reads the deterministic 203-cell Stephenson AP4
MuData fixture (recorded in ``data/manifest.yaml``, kept out of Git), applies
:func:`airr_adapter.select_heavy_chains`, and writes the two-column encoder
input CSV (``barcode,contigs,cdr3``) plus a JSON provenance sidecar with row
counts and a SHA-256 of the CSV.

The output goes under ``data/external/`` (gitignored) by default: the processed
Stephenson dataset's redistribution licence is still ``pending_verification`` in
the manifest, so derived receptor sequences must not be committed. Regenerate on
demand from the local fixture instead.

Usage (from repo root, in the benisse-scirpy022 env):

    python derive_ap4_encoder_input.py \
        --fixture data/external/stephenson2021_ap4_203.h5mu \
        --out data/external/ap4_encoder_input.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import airr_adapter as aa

DEFAULT_FIXTURE = Path("data/external/stephenson2021_ap4_203.h5mu")
DEFAULT_OUT = Path("data/external/ap4_encoder_input.csv")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def derive(fixture: Path, out: Path) -> dict:
    import mudata

    mdata = mudata.read_h5mu(fixture)
    heavy = aa.select_heavy_chains(mdata)
    aa.assert_reversible_barcodes(heavy.index)
    table = aa.write_encoder_input_csv(heavy, out)

    provenance = {
        "schema_version": aa.SCHEMA_VERSION,
        "source_fixture": str(fixture),
        "selection": "productive IGH, ranked by abundance then junction_aa then sequence_id",
        "rows": int(len(table)),
        "unique_junction_aa": int(table["cdr3"].nunique()),
        "output_csv": str(out),
        "output_sha256": _sha256(out),
    }
    sidecar = out.with_suffix(out.suffix + ".provenance.json")
    sidecar.write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if not args.fixture.exists():
        raise SystemExit(
            f"Fixture not found: {args.fixture}\n"
            "It is gitignored; see data/manifest.yaml to reproduce it locally."
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    provenance = derive(args.fixture, args.out)
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
