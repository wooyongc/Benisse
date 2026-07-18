import hashlib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
HASH_LEDGER = REPO_ROOT / "example" / "reference-output-hashes.sha256"


def test_committed_reference_hashes():
    for line in HASH_LEDGER.read_text().splitlines():
        expected, relative_path = line.split(maxsplit=1)
        path = REPO_ROOT / relative_path
        assert path.is_file(), relative_path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, relative_path
