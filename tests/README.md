# Scientific parity checks

Phase 4a adds tests around the legacy scientific contract without creating a public package
or stabilizing a new CLI.

Run the fast pytest checks in the `benisse-scirpy022` environment:

```sh
conda run -n benisse-scirpy022 python -m pytest -v
```

The fast suite checks the callable encoder, duplicate multi-file input, and every committed
reference hash. It runs entirely on CPU; the full pipeline test is skipped unless explicitly
enabled.

Run the complete Python + R oracle check explicitly:

```sh
BENISSE_RUN_SLOW_TESTS=1 conda run -n benisse-scirpy022 python -m pytest -m slow -v
```

The complete check takes several minutes. Stable CSV/text outputs are compared byte-for-byte,
`Benisse_results.RData` is compared semantically with exact sparse-edge agreement, and PDF
plots are rasterized with `pdftoppm` before byte comparison so timestamps do not cause false
failures.
