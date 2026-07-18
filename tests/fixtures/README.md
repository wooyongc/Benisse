# R-core golden fixture

`r_core_golden.json` is generated from a corrected, test-only R implementation that uses
one optimization variable per undirected edge and the corresponding symmetric-edge gradient.
It covers the eigendecomposition and dual updates, bounded L-BFGS-B graph update, latent
distances, corrected convergence metric, and a complete small ADMM run.

Regenerate it from the repository root with:

```sh
Rscript tests/fixtures/generate_r_core_golden.R tests/fixtures/r_core_golden.json
```

The fixture is deliberately independent of the Phase 4c Python-to-R bridge and AIRR
adapter. The component tests can therefore detect numerical-port drift without conflating
it with input conversion or subprocess behavior.

`corrected_update_A.R` is a fixture generator, not a replacement for production `R/update.R`.
Full-example and cohort-scale equivalence are explicitly deferred; the Python core remains
experimental and the Phase 4c R bridge remains the supported execution path.
