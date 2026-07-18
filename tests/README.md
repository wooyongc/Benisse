# Scientific parity checks

Phase 4a adds tests around the legacy scientific contract without creating a public package
or stabilizing a new CLI.

Run the fast pytest checks in the `benisse-scirpy022` environment:

```sh
conda run -n benisse-scirpy022 python -m pytest -v
```

The fast suite checks the callable encoder contract, duplicate and disjoint multi-file input,
input-order invariance, every committed reference hash, the corrected convergence norm, and
scientific invariants of the R oracle. It runs entirely on CPU; the full pipeline test is
skipped unless explicitly enabled.

Run the complete Python + R oracle check explicitly:

```sh
BENISSE_RUN_SLOW_TESTS=1 conda run -n benisse-scirpy022 python -m pytest -m slow -v
```

The complete check takes several minutes. Stable CSV/text outputs are compared byte-for-byte,
`Benisse_results.RData` is compared semantically with exact sparse-edge agreement, and PDF
plots are rasterized with `pdftoppm` before byte comparison so timestamps do not cause false
failures.

Despite its filename, `example/sparse_graph.txt` stores the weighted matrix `A`. The binary
adjacency used for exact edge-set parity is `results$sparse_graph` inside
`example/Benisse_results.RData`. Tests preserve and assert this distinction for downstream
AnnData and Python-port interfaces.
