#!/usr/bin/env python3
"""Run the complete legacy pipeline and compare it with the scientific oracle."""

import hashlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = REPO_ROOT / "example"
STABLE_R_OUTPUTS = (
    "cleaned_exp.txt",
    "clonality_label.txt",
    "clone_annotation.csv",
    "latent_dist.txt",
    "sparse_graph.txt",
)
PDF_OUTPUTS = ("connectionplot.pdf", "in_cross_dist_check.pdf")


def run(command, timeout=1800):
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(
        [str(part) for part in command],
        cwd=REPO_ROOT,
        check=True,
        timeout=timeout,
    )


def require_executable(name):
    executable = shutil.which(name)
    if executable is None:
        raise RuntimeError(f"Required parity tool is unavailable: {name}")
    return executable


def assert_same_bytes(actual, expected):
    actual_bytes = actual.read_bytes()
    expected_bytes = expected.read_bytes()
    if actual_bytes != expected_bytes:
        raise AssertionError(f"Byte mismatch: {actual} != {expected}")
    print(f"byte parity: {expected.name} ({hashlib.sha256(actual_bytes).hexdigest()})")


def verify_committed_hashes():
    ledger = EXAMPLE_DIR / "reference-output-hashes.sha256"
    for line in ledger.read_text().splitlines():
        expected_hash, relative_path = line.split(maxsplit=1)
        path = REPO_ROOT / relative_path
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise AssertionError(f"Reference hash mismatch: {relative_path}")
    print("committed reference hashes: OK")


def verify_rdata(actual):
    reference = EXAMPLE_DIR / "Benisse_results.RData"
    r_expression = f"""
reference <- new.env()
actual <- new.env()
load('{reference.as_posix()}', envir=reference)
load('{actual.as_posix()}', envir=actual)
comparison <- all.equal(reference$results, actual$results, check.attributes=TRUE)
if (!isTRUE(comparison)) stop(comparison)
if (!identical(reference$results$sparse_graph, actual$results$sparse_graph)) {{
  stop('Sparse edge set differs from the reference')
}}
cat('RData semantic parity: TRUE\\n')
cat('exact sparse edges: TRUE; matrix entries =',
    sum(actual$results$sparse_graph), '\\n')
"""
    run([require_executable("Rscript"), "-e", r_expression])


def render_pdf(pdf, output_prefix):
    run([require_executable("pdftoppm"), "-png", "-r", "120", pdf, output_prefix])
    pages = sorted(output_prefix.parent.glob(output_prefix.name + "-*.png"))
    if not pages:
        raise AssertionError(f"No pages rendered from {pdf}")
    return pages


def verify_pdfs(actual_dir, work_dir):
    for filename in PDF_OUTPUTS:
        reference_pages = render_pdf(
            EXAMPLE_DIR / filename,
            work_dir / (Path(filename).stem + "-reference"),
        )
        actual_pages = render_pdf(
            actual_dir / filename,
            work_dir / (Path(filename).stem + "-actual"),
        )
        if len(reference_pages) != len(actual_pages):
            raise AssertionError(f"PDF page-count mismatch: {filename}")
        for reference_page, actual_page in zip(reference_pages, actual_pages):
            assert_same_bytes(actual_page, reference_page)
        print(f"rendered PDF parity: {filename}")


def main():
    verify_committed_hashes()
    with tempfile.TemporaryDirectory(prefix="benisse-parity-") as tmpdir:
        work_dir = Path(tmpdir)
        encoded = work_dir / "encoded_10x_NSCLC.csv"
        r_output = work_dir / "r-output"
        r_output.mkdir()

        run(
            [
                sys.executable,
                REPO_ROOT / "AchillesEncoder.py",
                "--input_data",
                EXAMPLE_DIR / "10x_NSCLC.csv",
                "--output_data",
                encoded,
                "--cuda",
                "False",
            ]
        )
        assert_same_bytes(encoded, EXAMPLE_DIR / "encoded_10x_NSCLC.csv")

        run(
            [
                require_executable("Rscript"),
                REPO_ROOT / "Benisse.R",
                EXAMPLE_DIR / "10x_NSCLC_exp.csv",
                EXAMPLE_DIR / "10x_NSCLC_contigs.csv",
                encoded,
                r_output,
                "1610",
                "1",
                "100",
                "1",
                "1",
                "10",
                "1e-10",
            ]
        )

        for filename in STABLE_R_OUTPUTS:
            assert_same_bytes(r_output / filename, EXAMPLE_DIR / filename)
        verify_rdata(r_output / "Benisse_results.RData")
        verify_pdfs(r_output, work_dir)

    print("complete scientific parity: PASS")


if __name__ == "__main__":
    main()
