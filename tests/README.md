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
on CPU; cohort-scale validation of the experimental Python core is deferred.

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
