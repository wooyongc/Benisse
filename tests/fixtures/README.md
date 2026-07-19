# R-core golden fixture

`r_core_golden.json` is generated from a corrected, test-only R implementation that uses
one optimization variable per undirected edge and the corresponding symmetric-edge gradient.
It covers the eigendecomposition and dual updates, bounded L-BFGS-B graph update, latent
distances, corrected convergence metric, and a complete small ADMM run. It also records a
separate `legacy_production_gap` result from shipped `R/update.R`; regular tests assert that
frozen v1 retains edge `(0,2)` while corrected v2 removes it.

Regenerate it from the repository root with:

```sh
Rscript tests/fixtures/generate_r_core_golden.R tests/fixtures/r_core_golden.json
```

The fixture is deliberately independent of the Phase 4c Python-to-R bridge and AIRR
adapter. The component tests can therefore detect numerical-port drift without conflating
it with input conversion or subprocess behavior.

`corrected_update_A.R` is a fixture generator, not a rewrite of historical `R/update.R`.
Production R remains the frozen v1 oracle. Corrected Python is the v2 algorithm, so the known
edge-set gap is migration evidence rather than an equivalence failure.
