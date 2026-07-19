"""Phase 4c Python->R bridge tests.

Fast tests (always run) cover CLI-arg assembly, subprocess error handling, and
parsing R outputs into the 4b schema -- including parsing the committed
``example/`` reference outputs, which validates the parser against *real* R
oracle output without running R. The full encode+R end-to-end is a slow test,
skipped unless ``BENISSE_RUN_SLOW_TESTS=1`` and ``Rscript`` is available.
"""

import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import airr_adapter as aa  # noqa: E402
import benisse_bridge as bb  # noqa: E402

EXAMPLE = REPO_ROOT / "example"


# --- parameters and command assembly --------------------------------------


def test_params_default_cli_args_order_and_format():
    args = bb.BenisseRParams().as_cli_args()
    # lambda2, gamma, max_iter, lambda1, rho, m, stop_cutoff (example defaults)
    assert args == ["1610", "1", "100", "1", "1", "10", "1e-10"]


def test_fmt_keeps_integers_clean_and_rejects_bool():
    assert bb._fmt(100) == "100"
    assert bb._fmt(1.0) == "1"        # no spurious ".0"
    assert bb._fmt(1e-10) == "1e-10"
    with pytest.raises(TypeError):
        bb._fmt(True)


def test_build_command_positions_all_args():
    cmd = bb.build_benisse_command(
        "e.csv", "c.csv", "enc.csv", "/out", bb.BenisseRParams(lambda2=42),
        benisse_r="/repo/Benisse.R", rscript="Rscript",
    )
    assert cmd[:6] == ["Rscript", "/repo/Benisse.R", "e.csv", "c.csv", "enc.csv", "/out"]
    assert cmd[6] == "42"  # lambda2
    assert cmd[-1] == "1e-10"  # stop_cutoff


# --- subprocess error handling --------------------------------------------


def test_run_benisse_r_missing_input_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        bb.run_benisse_r(
            tmp_path / "nope_exp.csv", tmp_path / "nope_c.csv",
            tmp_path / "nope_enc.csv", tmp_path / "out",
        )


def test_run_benisse_r_propagates_nonzero_exit(tmp_path):
    # Real inputs so the existence checks pass, but a command that always fails.
    for name in ("exp.csv", "contigs.csv", "encoded.csv", "Benisse.R"):
        (tmp_path / name).write_text("x")
    with pytest.raises(RuntimeError, match="exit"):
        bb.run_benisse_r(
            tmp_path / "exp.csv", tmp_path / "contigs.csv", tmp_path / "encoded.csv",
            tmp_path / "out", benisse_r=tmp_path / "Benisse.R", rscript="/usr/bin/false",
        )


# --- parsing R output into the 4b schema ----------------------------------


def _write_fake_outputs(out_dir, A, clone_rows):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_dir / bb.OUTPUT_FILES["sparse_graph"], A)
    header = '"","v_gene","j_gene","cdr3","clsize","graph_label"\n'
    lines = [header]
    for i, (bc, v, j, c) in enumerate(clone_rows):
        lines.append(f'"{bc}","{v}","{j}","{c}",1,"single {i}"\n')
    (out_dir / bb.OUTPUT_FILES["clone_annotation"]).write_text("".join(lines))


def test_read_benisse_network_synthetic_upper_triangle(tmp_path):
    A = np.array([
        [0.0, 0.5, 0.0, 0.0],
        [0.5, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ])
    rows = [("bc0", "IGHV1", "IGHJ1", "CARAA"), ("bc1", "IGHV2", "IGHJ2", "CARBB"),
            ("bc2", "IGHV3", "IGHJ3", "CARCC"), ("bc3", "IGHV4", "IGHJ4", "CARDD")]
    _write_fake_outputs(tmp_path, A, rows)

    res = bb.read_benisse_network(tmp_path)
    assert res.node_ids == [
        "IGHV1_CARAA_IGHJ1", "IGHV2_CARBB_IGHJ2",
        "IGHV3_CARCC_IGHJ3", "IGHV4_CARDD_IGHJ4",
    ]
    assert res.directed is False
    assert res.n_edges == 2  # (0,1) and (1,2), upper triangle only
    assert np.all(res.row <= res.col)
    coo = res.to_coo().toarray()
    assert coo[0, 1] == 0.5 and coo[1, 2] == 1.0
    aa.validate_network_result(res)  # schema-valid


def test_read_benisse_network_rejects_asymmetric(tmp_path):
    A = np.array([[0.0, 0.5], [0.9, 0.0]])  # not symmetric
    _write_fake_outputs(tmp_path, A, [("b0", "V", "J", "C0"), ("b1", "V", "J", "C1")])
    with pytest.raises(ValueError, match="symmetric"):
        bb.read_benisse_network(tmp_path)


def test_read_benisse_network_rejects_dim_mismatch(tmp_path):
    A = np.zeros((3, 3))
    _write_fake_outputs(tmp_path, A, [("b0", "V", "J", "C0"), ("b1", "V", "J", "C1")])
    with pytest.raises(ValueError, match="!="):
        bb.read_benisse_network(tmp_path)


def test_read_benisse_network_no_edges(tmp_path):
    A = np.zeros((3, 3))
    rows = [("b0", "V1", "J1", "C0"), ("b1", "V2", "J2", "C1"), ("b2", "V3", "J3", "C2")]
    _write_fake_outputs(tmp_path, A, rows)
    res = bb.read_benisse_network(tmp_path)
    assert res.n_nodes == 3 and res.n_edges == 0
    aa.validate_network_result(res)


def test_read_benisse_network_single_node(tmp_path):
    # np.loadtxt returns a 0-d scalar for a 1x1 matrix; must still parse.
    _write_fake_outputs(tmp_path, np.zeros((1, 1)), [("b0", "V", "J", "C0")])
    res = bb.read_benisse_network(tmp_path)
    assert res.n_nodes == 1 and res.n_edges == 0
    assert res.node_ids == ["V_C0_J"]


def test_read_benisse_network_rejects_nonzero_diagonal(tmp_path):
    A = np.array([[5.0, 0.0], [0.0, 0.0]])  # symmetric, finite, self-loop
    _write_fake_outputs(tmp_path, A, [("b0", "V", "J", "C0"), ("b1", "V", "J", "C1")])
    with pytest.raises(ValueError, match="diagonal"):
        bb.read_benisse_network(tmp_path)


def test_read_benisse_network_rejects_non_finite(tmp_path):
    A = np.array([[0.0, np.nan], [np.nan, 0.0]])  # structurally symmetric NaN
    _write_fake_outputs(tmp_path, A, [("b0", "V", "J", "C0"), ("b1", "V", "J", "C1")])
    with pytest.raises(ValueError, match="non-finite"):
        bb.read_benisse_network(tmp_path)


def test_read_benisse_network_rejects_duplicate_clone_keys(tmp_path):
    A = np.zeros((2, 2))
    # identical v/j/cdr3 -> identical clone keys -> non-unique node_ids
    _write_fake_outputs(tmp_path, A, [("b0", "V", "J", "SAME"), ("b1", "V", "J", "SAME")])
    with pytest.raises(ValueError, match="unique"):
        bb.read_benisse_network(tmp_path)


def test_run_benisse_r_missing_rscript(tmp_path):
    for name in ("exp.csv", "contigs.csv", "encoded.csv", "Benisse.R"):
        (tmp_path / name).write_text("x")
    with pytest.raises(FileNotFoundError, match="Rscript"):
        bb.run_benisse_r(
            tmp_path / "exp.csv", tmp_path / "contigs.csv", tmp_path / "encoded.csv",
            tmp_path / "out", benisse_r=tmp_path / "Benisse.R",
            rscript="definitely-not-a-real-binary-xyz",
        )


# --- real oracle: parse the committed example reference outputs ------------


@pytest.mark.skipif(
    not (EXAMPLE / "sparse_graph.txt").exists(),
    reason="committed example reference outputs not present",
)
def test_parse_committed_example_reference_outputs():
    res = bb.read_benisse_network(EXAMPLE)
    # Ground truth measured from the committed reference outputs.
    assert res.n_nodes == 1494
    assert res.n_edges == 1691          # upper-triangle edges of the symmetric A
    assert len(set(res.node_ids)) == res.n_nodes  # clone keys unique
    assert res.weight.min() >= 0.5 and res.weight.max() <= 1.0
    assert np.all(res.row < res.col)    # strictly upper-triangular (no self-loops)
    aa.validate_network_result(res)


# --- slow end-to-end: encode + run the unmodified R core ------------------


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("BENISSE_RUN_SLOW_TESTS") != "1",
    reason="set BENISSE_RUN_SLOW_TESTS=1 to run the multi-minute R core",
)
@pytest.mark.skipif(shutil.which("Rscript") is None, reason="Rscript not available")
def test_run_pipeline_matches_reference_edge_set(tmp_path):
    result = bb.run_pipeline(
        EXAMPLE / "10x_NSCLC.csv",
        EXAMPLE / "10x_NSCLC_exp.csv",
        EXAMPLE / "10x_NSCLC_contigs.csv",
        tmp_path / "run",
        bb.BenisseRParams(),  # example defaults (lambda2=1610)
        cuda=False,
    )
    reference = bb.read_benisse_network(EXAMPLE)
    assert result.node_ids == reference.node_ids
    got = set(zip(result.row.tolist(), result.col.tolist()))
    want = set(zip(reference.row.tolist(), reference.col.tolist()))
    assert got == want  # exact discrete edge-set agreement with the R oracle
