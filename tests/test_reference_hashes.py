import hashlib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
HASH_LEDGER = REPO_ROOT / "example" / "reference-output-hashes.sha256"
EXPECTED_REFERENCES = {
    "example/encoded_10x_NSCLC.csv",
    "example/Benisse_results.RData",
    "example/cleaned_exp.txt",
    "example/clonality_label.txt",
    "example/clone_annotation.csv",
    "example/connectionplot.pdf",
    "example/in_cross_dist_check.pdf",
    "example/latent_dist.txt",
    "example/sparse_graph.txt",
}


def test_committed_reference_hashes():
    entries = [line.split(maxsplit=1) for line in HASH_LEDGER.read_text().splitlines()]
    assert {relative_path for _, relative_path in entries} == EXPECTED_REFERENCES

    for expected, relative_path in entries:
        path = REPO_ROOT / relative_path
        assert path.is_file(), relative_path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, relative_path
