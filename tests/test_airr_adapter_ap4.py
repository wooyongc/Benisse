"""Phase 4b parity checks against the real Stephenson AP4 MuData fixture.

These run only when the gitignored fixture is present locally; otherwise they
skip (CI has no access to it, and its redistribution licence is unconfirmed).
They validate the adapter against the ecosystem's own object: our independent
heavy-chain selection must agree with scirpy's ``chain_indices`` primary VDJ
pointer, and the derived counts must match ``data/manifest.yaml``.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import airr_adapter as aa  # noqa: E402

FIXTURE = REPO_ROOT / "data" / "external" / "stephenson2021_ap4_203.h5mu"

mudata = pytest.importorskip("mudata")
pytestmark = pytest.mark.skipif(
    not FIXTURE.exists(), reason=f"AP4 fixture not present: {FIXTURE}"
)


@pytest.fixture(scope="module")
def ap4():
    return mudata.read_h5mu(FIXTURE)


def test_selection_matches_manifest_shape(ap4):
    heavy = aa.select_heavy_chains(ap4)
    # manifest: cells_with_productive_heavy_and_light == 203
    assert len(heavy) == 203
    assert (heavy["locus"] == "IGH").all()
    assert heavy["productive"].astype(bool).all()
    assert heavy["junction_aa"].astype("string").notna().all()


def test_selection_agrees_with_scirpy_primary_vdj(ap4):
    """Our deterministic ranking == scirpy chain_indices VDJ[0] on every cell."""
    heavy = aa.select_heavy_chains(ap4)
    vdj0 = aa.scirpy_primary_heavy_index(ap4)
    assert vdj0 is not None

    airr = ap4.mod["airr"].obsm["airr"]
    cell_ids = list(map(str, ap4.mod["airr"].obs_names))
    pos_of = {cid: i for i, cid in enumerate(cell_ids)}

    checked = 0
    for cid in heavy.index:
        pos = vdj0.loc[cid]
        if pos is None or (isinstance(pos, float) and np.isnan(pos)):
            continue
        scirpy_seq = str(airr[pos_of[cid]][int(pos)]["sequence_id"])
        assert heavy.loc[cid, "sequence_id"] == scirpy_seq
        checked += 1
    assert checked == 203


def test_manifest_airr_summary_counts(ap4):
    long = aa.airr_to_dataframe(ap4)
    igh = long[long["locus"] == "IGH"]
    prod = igh[
        igh["productive"].fillna(False).astype(bool)
        & igh["junction_aa"].astype("string").notna()
    ]
    # manifest airr_summary
    assert len(prod) == 216  # productive_igh_with_junction_aa
    assert prod["junction_aa"].nunique() == 214  # unique_productive_igh_junction_aa
    assert long["locus"].value_counts().to_dict() == {"IGH": 233, "IGK": 138, "IGL": 121}


def test_barcodes_are_r_join_reversible(ap4):
    # Real barcodes (e.g. "S15_AAAC...-1") must survive '-'<->'.' normalisation.
    aa.assert_reversible_barcodes(map(str, ap4.mod["airr"].obs_names))


def test_encoder_input_maps_junction_aa_with_conserved_residues(ap4):
    heavy = aa.select_heavy_chains(ap4)
    table = aa.build_encoder_input(heavy)
    assert list(table.columns) == ["contigs", "cdr3"]
    # AIRR junction_aa keeps the conserved C..W/F boundary residues, matching the
    # committed encoder example (e.g. "CAK...W"); it is NOT the narrower cdr3_aa.
    assert (table["cdr3"].str.startswith("C")).mean() > 0.9
    # contigs are the AIRR sequence_id (unique per selected chain).
    assert table["contigs"].is_unique


def test_embedding_write_preserves_airr_modality(ap4):
    import pandas as pd

    mdata = ap4  # module-scoped; we add a new obsm key without touching airr
    heavy = aa.select_heavy_chains(mdata)
    emb = pd.DataFrame(
        np.zeros((len(heavy), 20), dtype="float64"), index=heavy.index
    )
    fields_before = list(mdata.mod["airr"].obsm["airr"].fields)
    n_chain_idx = len(mdata.mod["airr"].obsm["chain_indices"])

    aa.attach_embedding(mdata, emb, modality="airr", key="X_benisse_test")

    assert mdata.mod["airr"].obsm["X_benisse_test"].shape == (mdata.n_obs, 20)
    assert list(mdata.mod["airr"].obsm["airr"].fields) == fields_before
    assert len(mdata.mod["airr"].obsm["chain_indices"]) == n_chain_idx
