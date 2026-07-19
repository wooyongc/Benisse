# Scientific parity checks

Phase 4a adds tests around the legacy scientific contract without creating a public package
or stabilizing a new CLI.

Run the fast pytest checks in the `benisse-scirpy022` environment:

```sh
conda run -n benisse-scirpy022 python -m pytest -v
```

The fast suite checks the callable encoder contract, duplicate and disjoint multi-file input,
input-order invariance, every committed reference hash, the corrected convergence norm, and
scientific invariants of the R oracle. Phase 4d adds small corrected-R fixtures for each
NumPy/SciPy kernel, finite-difference gradient checks, optimizer-status enforcement, a complete
four-node ADMM comparison, permutation and alternate-hyperparameter cases, and a configuration
test for the `n > 1000` optimizer branch without allocating a large matrix. It runs entirely
on CPU. Guarded sample-level and committed-example validation of the experimental Python core
is described in `PHASE4D_NOTES.md`.

Regenerate and inspect the Phase 4d component oracle with:

```sh
Rscript tests/fixtures/generate_r_core_golden.R tests/fixtures/r_core_golden.json
conda run -n benisse-scirpy022 python -m pytest tests/test_python_r_core.py -v
```

Run the complete **legacy R pipeline** oracle check explicitly:

```sh
BENISSE_RUN_SLOW_TESTS=1 conda run -n benisse-scirpy022 python -m pytest -m slow -v
```

This check does not validate the experimental Python core. It takes several minutes; stable
CSV/text outputs are compared byte-for-byte,
`Benisse_results.RData` is compared semantically with exact sparse-edge agreement, and PDF
plots are rasterized with `pdftoppm` before byte comparison so timestamps do not cause false
failures.

Despite its filename, `example/sparse_graph.txt` stores the weighted matrix `A`. The binary
adjacency used for exact edge-set parity is `results$sparse_graph` inside
`example/Benisse_results.RData`. Tests preserve and assert this distinction for downstream
AnnData and Python-port interfaces.

## Local real-data validation

The regular suite does not require downloaded data. If the local Stephenson MuData object is
present, run the complete AP4 sample validation explicitly:

```sh
BENISSE_RUN_LOCAL_DATA_TESTS=1 conda run -n benisse-scirpy022 \
  python -m pytest tests/test_real_data_validation.py \
  -k ap4_complete_sample_validation -v
```

Run the corrected core on the committed NSCLC example explicitly with:

```sh
BENISSE_RUN_PYTHON_CORE_EXAMPLE=1 conda run -n benisse-scirpy022 \
  python -m pytest tests/test_real_data_validation.py \
  -k corrected_core_on_committed_nsclc_example -v
```

For other Stephenson validations, invoke `real_data_validation.py` with one `--sample-id` and
write outputs under `/tmp`. Do not submit the entire 5k multi-patient object as one graph; the
tool refuses more than 500 clone nodes unless the guard is deliberately raised.

## Python post-analysis parity

Run the Phase 2 scientific and rendering checks with:

```sh
conda run -n benisse-scirpy022 python -m pytest tests/test_benisse_plotting.py -v
```

When R is available, this exports `master_dist_e`, the V/J candidate graph, and the two coupling
correlations from the committed `Benisse_results.RData` into pytest's temporary directory. The
Python expression-distance matrix and correlation statistics are then compared directly with
that oracle. Synthetic checks cover sparse and empty graphs and verify that every Python plot
writes a nonempty PDF. No generated plot is committed over the reference R PDFs.
