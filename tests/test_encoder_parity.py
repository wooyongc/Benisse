import argparse
import hashlib
import shutil
from pathlib import Path

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
    assert output.read_bytes() == REFERENCE_ENCODING.read_bytes()
    assert sha256(output) == (
        "400eb1b07fe436779d138c494ca28bdb7c0cb630ef1e05cb5047c62fde56605d"
    )


def test_multifile_encoder_matches_single_file_reference(tmp_path):
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    shutil.copyfile(EXAMPLE_INPUT, input_dir / "a.csv")
    shutil.copyfile(EXAMPLE_INPUT, input_dir / "b.csv")
    output = tmp_path / "encoded.csv"

    encode_bcr(input_dir, output, cuda=False)

    assert output.read_bytes() == REFERENCE_ENCODING.read_bytes()
