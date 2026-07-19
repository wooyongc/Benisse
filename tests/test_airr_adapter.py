"""Phase 4b AIRR adapter tests on synthetic scverse objects.

These never touch external data, so they always run. They pin the deterministic
contract of ``airr_adapter``: heavy-chain selection and its tie-breaks, barcode
join reversibility, encoder-input field mapping, embedding preservation, and the
clonotype-network result schema. The AP4 fixture provides the real-object parity
check separately (``test_airr_adapter_ap4.py``).
"""

import sys
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import airr_adapter as aa  # noqa: E402

anndata = pytest.importorskip("anndata")
mudata = pytest.importorskip("mudata")


# --- synthetic object builders -------------------------------------------

_FIELDS = (
    "locus",
    "productive",
    "junction_aa",
    "sequence_id",
    "duplicate_count",
    "consensus_count",
    "v_call",
    "j_call",
    "c_call",
)


def _chain(locus, junction_aa, sequence_id, productive=True, dup=1, cons=1,
           v="IGHV1", j="IGHJ1", c="IGHM"):
    return {
        "locus": locus,
        "productive": productive,
        "junction_aa": junction_aa,
        "sequence_id": sequence_id,
        "duplicate_count": dup,
        "consensus_count": cons,
        "v_call": v,
        "j_call": j,
        "c_call": c,
    }


def _airr_anndata(cells):
    """cells: list of (barcode, [chain dicts]) -> AnnData with obsm['airr']."""
    barcodes = [b for b, _ in cells]
    records = [chains for _, chains in cells]
    airr = ak.Array(records)
    adata = anndata.AnnData(np.zeros((len(barcodes), 0), dtype="float32"))
    adata.obs_names = barcodes
    adata.obsm["airr"] = airr
    return adata


# A: two IGH tie on count -> junction_aa breaks tie ("CARAA" < "CARWY").
# B: productive IGH beats higher-count non-productive IGH.
# C: light chains only -> dropped.
# D: empty junction_aa IGH dropped in favour of the one with a junction.
_CELLS = [
    ("cell_A-1", [
        _chain("IGH", "CARWY", "A_h1", dup=10, cons=100),
        _chain("IGK", "CQQAA", "A_l1", dup=8, cons=80),
        _chain("IGH", "CARAA", "A_h2", dup=10, cons=50),
    ]),
    ("cell_B-1", [
        _chain("IGH", "CARBIG", "B_h_np", productive=False, dup=99, cons=99),
        _chain("IGH", "CARSMALL", "B_h_ok", productive=True, dup=5, cons=5),
    ]),
    ("cell_C-1", [
        _chain("IGK", "CQQCC", "C_l1", dup=7, cons=70),
    ]),
    ("cell_D-1", [
        _chain("IGH", "", "D_h_empty", dup=50, cons=50),
        _chain("IGH", "CARDD", "D_h_ok", dup=3, cons=3),
    ]),
]


def test_heavy_selection_determinism_and_filters():
    adata = _airr_anndata(_CELLS)
    heavy = aa.select_heavy_chains(adata)

    # cell_C has no heavy chain and is dropped; the rest each yield one row.
    assert list(heavy.index) == ["cell_A-1", "cell_B-1", "cell_D-1"]
    # Tie on count broken by ascending junction_aa.
    assert heavy.loc["cell_A-1", "sequence_id"] == "A_h2"
    assert heavy.loc["cell_A-1", "junction_aa"] == "CARAA"
    # Productive filter beats raw abundance.
    assert heavy.loc["cell_B-1", "sequence_id"] == "B_h_ok"
    # Empty junction dropped.
    assert heavy.loc["cell_D-1", "sequence_id"] == "D_h_ok"


def test_heavy_selection_independent_of_chain_order():
    forward = aa.select_heavy_chains(_airr_anndata(_CELLS))
    reversed_cells = [(b, list(reversed(chains))) for b, chains in _CELLS]
    backward = aa.select_heavy_chains(_airr_anndata(reversed_cells))
    pd.testing.assert_frame_equal(forward, backward)


def test_heavy_selection_rejects_missing_or_ambiguous_gene_calls_before_ranking():
    cells = [
        ("missing-1", [
            _chain("IGH", "CARBAD", "bad", dup=99, v=None, j=None),
            _chain("IGH", "CARGOOD", "good", dup=1, v="IGHV1*01", j="IGHJ2*02"),
        ]),
        ("ambiguous-1", [
            _chain("IGH", "CARAMBIG", "ambig", v="IGHV1*01,IGHV2*01", j="IGHJ1"),
        ]),
        ("alleles-1", [
            _chain(
                "IGH", "CARALLELE", "alleles",
                v="IGHV3-23*01,IGHV3-23*02", j="IGHJ4*02",
            ),
        ]),
    ]
    heavy = aa.select_heavy_chains(_airr_anndata(cells))

    assert list(heavy.index) == ["missing-1", "alleles-1"]
    assert heavy.loc["missing-1", "sequence_id"] == "good"
    assert heavy.loc["missing-1", "v_call"] == "IGHV1"
    assert heavy.loc["missing-1", "j_call"] == "IGHJ2"
    assert heavy.loc["alleles-1", "v_call"] == "IGHV3-23"
    assert heavy.loc["alleles-1", "j_call"] == "IGHJ4"


def test_count_fallback_to_consensus_when_no_duplicate_count():
    # Two productive IGH, equal on duplicate_count; consensus is only a later
    # tiebreak, so junction_aa still decides here -- assert count coalescing
    # picks duplicate_count first (both 4 -> tie -> junction_aa).
    cells = [("z-1", [
        _chain("IGH", "CARZB", "z2", dup=4, cons=999),
        _chain("IGH", "CARZA", "z1", dup=4, cons=1),
    ])]
    heavy = aa.select_heavy_chains(_airr_anndata(cells))
    assert heavy.loc["z-1", "sequence_id"] == "z1"  # CARZA < CARZB
    assert heavy.loc["z-1", "selection_count"] == 4.0


def test_scirpy_primary_heavy_index_reads_chain_indices():
    adata = _airr_anndata(_CELLS[:1])
    adata.obsm["chain_indices"] = ak.Array(
        [{"VJ": [1, None], "VDJ": [0, None], "multichain": False}]
    )
    idx = aa.scirpy_primary_heavy_index(adata)
    assert idx is not None
    assert idx.loc["cell_A-1"] == 0


def test_scirpy_primary_heavy_index_absent_returns_none():
    assert aa.scirpy_primary_heavy_index(_airr_anndata(_CELLS)) is None


# --- barcode joining ------------------------------------------------------


def test_barcode_roundtrip():
    bc = "S15_AAACGGGGTACAGTTC-1"
    assert aa.to_r_barcode(bc) == "S15_AAACGGGGTACAGTTC.1"
    assert aa.from_r_barcode(aa.to_r_barcode(bc)) == bc


def test_barcode_reversibility_guard():
    aa.assert_reversible_barcodes(["a-1", "b-2", "c_x-1"])
    with pytest.raises(ValueError):
        aa.assert_reversible_barcodes(["already.dotted-1"])
    with pytest.raises(ValueError):
        # "a.1" already contains '.', so the dot-guard rejects the batch (this is
        # what makes '-'<->'.' non-invertible), before any injectivity check.
        aa.assert_reversible_barcodes(["a-1", "a.1"])


# --- encoder input mapping ------------------------------------------------


def test_build_encoder_input_columns_and_mapping():
    heavy = aa.select_heavy_chains(_airr_anndata(_CELLS))
    table = aa.build_encoder_input(heavy)
    assert list(table.columns) == ["contigs", "cdr3"]
    assert table.index.name == "barcode"
    # contigs <- sequence_id, cdr3 <- junction_aa
    assert table.loc["cell_A-1", "contigs"] == "A_h2"
    assert table.loc["cell_A-1", "cdr3"] == "CARAA"


def test_encoder_input_csv_layout_matches_example(tmp_path):
    heavy = aa.select_heavy_chains(_airr_anndata(_CELLS))
    path = tmp_path / "enc.csv"
    aa.write_encoder_input_csv(heavy, path)
    header = path.read_text().splitlines()[0]
    assert header == "barcode,contigs,cdr3"  # same shape as example/10x_NSCLC.csv


# --- embedding preservation ----------------------------------------------


def test_attach_embedding_preserves_airr_and_aligns_rows():
    adata = _airr_anndata(_CELLS)
    heavy = aa.select_heavy_chains(adata)
    emb = pd.DataFrame(
        np.arange(len(heavy) * 3, dtype="float64").reshape(-1, 3),
        index=heavy.index,
    )
    fields_before = list(adata.obsm["airr"].fields)

    aa.attach_embedding(adata, emb, key="X_benisse")

    assert adata.obsm["X_benisse"].shape == (adata.n_obs, 3)
    assert list(adata.obsm["airr"].fields) == fields_before  # AIRR intact
    # cell_C had no heavy chain -> its embedding row is all NaN.
    c_pos = list(adata.obs_names).index("cell_C-1")
    assert np.all(np.isnan(adata.obsm["X_benisse"][c_pos]))
    a_pos = list(adata.obs_names).index("cell_A-1")
    assert not np.any(np.isnan(adata.obsm["X_benisse"][a_pos]))


def test_attach_embedding_on_mudata_leaves_gex_untouched():
    airr_ad = _airr_anndata(_CELLS)
    gex = anndata.AnnData(np.ones((airr_ad.n_obs, 4), dtype="float32"))
    gex.obs_names = list(airr_ad.obs_names)
    mdata = mudata.MuData({"gex": gex, "airr": airr_ad})

    heavy = aa.select_heavy_chains(mdata)
    emb = pd.DataFrame(np.ones((len(heavy), 2)), index=heavy.index)
    aa.attach_embedding(mdata, emb, modality="airr", key="X_benisse")

    assert "X_benisse" in mdata.mod["airr"].obsm
    assert "X_benisse" not in mdata.mod["gex"].obsm
    assert np.array_equal(mdata.mod["gex"].X, np.ones((airr_ad.n_obs, 4), dtype="float32"))


# --- clonotype-network result schema -------------------------------------


def _sample_result():
    return aa.BenisseNetworkResult(
        node_ids=["A_h2", "B_h_ok", "D_h_ok"],
        row=np.array([0, 1]),
        col=np.array([1, 2]),
        weight=np.array([0.5, 0.9]),
        params={"lambda": 1.0, "beta": 100},
        provenance=aa.benisse_provenance(model="trained_model.pt", schema=aa.SCHEMA_VERSION),
    )


def test_network_result_roundtrip_and_coo():
    res = _sample_result()
    aa.validate_network_result(res)
    coo = res.to_coo()
    assert coo.shape == (3, 3)
    assert coo.nnz == 2
    assert coo.toarray()[0, 1] == 0.5

    restored = aa.BenisseNetworkResult.from_dict(res.to_dict())
    assert restored.node_ids == res.node_ids
    assert np.array_equal(restored.weight, res.weight)


def test_network_result_validation_errors():
    dup = aa.BenisseNetworkResult(
        node_ids=["x", "x"], row=np.array([], int), col=np.array([], int),
        weight=np.array([], float),
    )
    with pytest.raises(ValueError):
        aa.validate_network_result(dup)

    oob = aa.BenisseNetworkResult(
        node_ids=["x", "y"], row=np.array([0]), col=np.array([5]),
        weight=np.array([1.0]),
    )
    with pytest.raises(ValueError):
        aa.validate_network_result(oob)


def test_attach_and_read_network_result_uses_uns_not_obsp():
    adata = _airr_anndata(_CELLS)
    res = _sample_result()
    aa.attach_network_result(adata, res)

    # Stored in uns with an explicit node index, never in cell-aligned obsp.
    assert "clonotype_network" in adata.uns["benisse"]
    assert len(adata.obsp) == 0

    read = aa.read_network_result(adata)
    assert read.node_ids == res.node_ids
    assert np.array_equal(read.row, res.row)
    assert read.params == res.params


# --- audit regression tests ----------------------------------------------


def test_string_valued_productive_is_parsed_not_truthy_cast():
    # AIRR TSV encodes booleans as "T"/"F"; bool("F") is True, so a naive cast
    # would keep the non-productive chain. The "F" chain must be dropped.
    cells = [("s-1", [
        _chain("IGH", "CARBIG", "s_np", productive="F", dup=99, cons=99),
        _chain("IGH", "CARSMALL", "s_ok", productive="T", dup=1, cons=1),
    ])]
    heavy = aa.select_heavy_chains(_airr_anndata(cells))
    assert heavy.loc["s-1", "sequence_id"] == "s_ok"


def test_count_coalesces_when_preferred_column_is_null():
    # duplicate_count present but null for every row -> fall back to
    # consensus_count, which then dominates the ranking over junction_aa.
    cells = [("q-1", [
        _chain("IGH", "CARQA", "q1", dup=None, cons=5),
        _chain("IGH", "CARQB", "q2", dup=None, cons=50),
    ])]
    heavy = aa.select_heavy_chains(_airr_anndata(cells))
    assert heavy.loc["q-1", "sequence_id"] == "q2"  # cons 50 > 5 beats CARQA<CARQB
    assert heavy.loc["q-1", "selection_count"] == 50.0


def test_missing_required_field_raises_clear_error():
    airr = ak.Array([[{"locus": "IGH", "productive": True, "junction_aa": "CARX"}]])
    adata = anndata.AnnData(np.zeros((1, 0), dtype="float32"))
    adata.obs_names = ["m-1"]
    adata.obsm["airr"] = airr
    with pytest.raises(ValueError, match="sequence_id"):
        aa.select_heavy_chains(adata)


def test_selection_deterministic_with_duplicate_sequence_id():
    # Non-conformant input (duplicate sequence_id, identical sort keys): the
    # storage-position tiebreak keeps the result deterministic for a given
    # object (picks chain_pos 0), though not order-independent.
    cells = [("d-1", [
        _chain("IGH", "CARSAME", "dupsid", dup=5, cons=5, c="IGHM"),
        _chain("IGH", "CARSAME", "dupsid", dup=5, cons=5, c="IGHG"),
    ])]
    obj = _airr_anndata(cells)
    first = aa.select_heavy_chains(obj)
    second = aa.select_heavy_chains(_airr_anndata(cells))
    assert len(first) == 1
    assert first.loc["d-1", "c_call"] == "IGHM"  # chain_pos 0
    pd.testing.assert_frame_equal(first, second)


def test_attach_embedding_rejects_barcode_namespace_mismatch():
    adata = _airr_anndata(_CELLS)
    # R-dotted barcodes match no obs_name -> must raise, not silently all-NaN.
    emb = pd.DataFrame(np.ones((1, 3)), index=["cell_A.1"])
    with pytest.raises(ValueError, match="namespace"):
        aa.attach_embedding(adata, emb)
    assert "X_benisse" not in adata.obsm


def test_attach_embedding_warns_on_partial_coverage():
    adata = _airr_anndata(_CELLS)
    emb = pd.DataFrame(np.ones((2, 3)), index=["cell_A-1", "ghost-1"])
    with pytest.warns(UserWarning, match="did not match"):
        aa.attach_embedding(adata, emb)
    assert adata.obsm["X_benisse"].shape == (adata.n_obs, 3)


def test_attach_embedding_refuses_reserved_keys():
    adata = _airr_anndata(_CELLS)
    emb = pd.DataFrame(np.ones((1, 3)), index=["cell_A-1"])
    for reserved in ("airr", "chain_indices"):
        with pytest.raises(ValueError, match="reserved"):
            aa.attach_embedding(adata, emb, key=reserved)


def test_validate_rejects_nan_weight_and_float_indices():
    nan_w = aa.BenisseNetworkResult(
        node_ids=["a", "b"], row=np.array([0]), col=np.array([1]),
        weight=np.array([np.nan]),
    )
    with pytest.raises(ValueError, match="NaN or infinite"):
        aa.validate_network_result(nan_w)

    float_idx = aa.BenisseNetworkResult(
        node_ids=["a", "b"], row=np.array([0.0]), col=np.array([1.0]),
        weight=np.array([1.0]),
    )
    with pytest.raises(ValueError, match="integer dtype"):
        aa.validate_network_result(float_idx)


def test_validate_enforces_undirected_upper_triangle():
    lower = aa.BenisseNetworkResult(
        node_ids=["a", "b"], row=np.array([1]), col=np.array([0]),
        weight=np.array([1.0]),
    )
    with pytest.raises(ValueError, match="upper-triangular"):
        aa.validate_network_result(lower)

    # Same edge is fine once marked directed, and round-trips through to_dict.
    directed = aa.BenisseNetworkResult(
        node_ids=["a", "b"], row=np.array([1]), col=np.array([0]),
        weight=np.array([1.0]), directed=True,
    )
    aa.validate_network_result(directed)
    assert aa.BenisseNetworkResult.from_dict(directed.to_dict()).directed is True
