import argparse
import hashlib
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from AchillesEncoder import encode_bcr, strtobool


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_INPUT = REPO_ROOT / "example" / "10x_NSCLC.csv"
REFERENCE_ENCODING = REPO_ROOT / "example" / "encoded_10x_NSCLC.csv"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scientific_encoding(path):
    """Read the CDR3-to-embedding mapping without the incidental CSV row index."""
    return (
        pd.read_csv(path, index_col=0)
        .sort_values("index", kind="mergesort")
        .reset_index(drop=True)
    )


def test_boolean_parser():
    assert strtobool("TRUE") is True
    assert strtobool("false") is False
    with pytest.raises(argparse.ArgumentTypeError):
        strtobool("yes")


def test_callable_encoder_matches_reference_bytes(tmp_path):
    output = tmp_path / "encoded.csv"
    encoded = encode_bcr(EXAMPLE_INPUT, output, cuda=False)

    reference_rows = len(REFERENCE_ENCODING.read_text().splitlines()) - 1
    assert len(encoded) == reference_rows
    assert list(encoded.columns) == list(range(20)) + ["index"]
    assert encoded["index"].is_unique
    assert encoded["index"].notna().all()
    assert np.isfinite(encoded[list(range(20))].to_numpy()).all()
    assert output.read_bytes() == REFERENCE_ENCODING.read_bytes()
    assert sha256(output) == (
        "400eb1b07fe436779d138c494ca28bdb7c0cb630ef1e05cb5047c62fde56605d"
    )


def test_duplicate_multifile_encoder_matches_single_file_reference(tmp_path):
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    shutil.copyfile(EXAMPLE_INPUT, input_dir / "a.csv")
    shutil.copyfile(EXAMPLE_INPUT, input_dir / "b.csv")
    output = tmp_path / "encoded.csv"

    encode_bcr(input_dir, output, cuda=False)

    assert output.read_bytes() == REFERENCE_ENCODING.read_bytes()


@pytest.mark.parametrize("reverse_files", [False, True])
def test_disjoint_multifile_encoder_concatenates_all_rows(tmp_path, reverse_files):
    frame = pd.read_csv(EXAMPLE_INPUT)
    split_at = len(frame) // 2
    first, second = frame.iloc[:split_at], frame.iloc[split_at:]
    if reverse_files:
        first, second = second, first

    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    first.to_csv(input_dir / "a.csv", index=False)
    second.to_csv(input_dir / "b.csv", index=False)
    output = tmp_path / "encoded.csv"

    encode_bcr(input_dir, output, cuda=False)

    pd.testing.assert_frame_equal(
        scientific_encoding(output),
        scientific_encoding(REFERENCE_ENCODING),
        check_exact=True,
    )


def test_encoder_is_invariant_to_input_row_order(tmp_path):
    frame = pd.read_csv(EXAMPLE_INPUT).iloc[::-1]
    reversed_input = tmp_path / "reversed.csv"
    frame.to_csv(reversed_input, index=False)
    output = tmp_path / "encoded.csv"

    encode_bcr(reversed_input, output, cuda=False)

    pd.testing.assert_frame_equal(
        scientific_encoding(output),
        scientific_encoding(REFERENCE_ENCODING),
        check_exact=True,
    )
