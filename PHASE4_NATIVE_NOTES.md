# R-free corrected Python pipeline — v2 migration record

Status: **the corrected Python implementation is the v2 default**. Runtime execution no longer
requires R. The historical `Benisse.R` workflow and Phase 4c bridge remain in the repository as
a frozen v1/paper oracle for regression and migration analysis; they are not the v2 backend.

Packaging, a stable public API, and a final CLI remain deferred until this scientific workflow
has received review.

## Runtime path

`benisse_pipeline.py` now provides two internal entry points:

- `run_csv_pipeline(...)`: encoder-input CSV + expression CSV + 10x contigs → encoder → Python
  preprocessing/initialization → corrected ADMM → `BenisseNetworkResult` → Python plots.
- `run_mudata_pipeline(...)`: MuData/AIRR + GEX → deterministic productive-heavy selection →
  encoder → corrected pipeline; attaches the 20-dimensional cell embedding to
  `mdata.mod["airr"].obsm["X_benisse"]` and the clone network to `mdata.uns["benisse"]`.

Both return the in-memory network, prepared matrices, complete ADMM diagnostics, latent
distances, annotation, compatible text/CSV outputs, and generated plot paths. Output provenance
states `implementation="corrected_python_v2"` and `runtime_requires_r=false`, and records hashes
for the ordered scientific inputs, available source files, and encoder checkpoint together with
software versions and the Git revision/dirty state.

The AIRR route accepts only productive heavy chains with non-empty, unambiguous V and J calls.
Allele-only alternatives (for example `IGHV3-23*01,IGHV3-23*02`) normalize to their common gene;
missing or multi-gene calls are excluded before abundance ranking and can never create a shared
empty-string candidate family. The selected cell-to-node assignment is persisted as
`airr.obs["benisse_clone_id"]`, including clonally expanded cells, and is H5MU-round-trip tested.
Awkward remains an AIRR-only dependency: standard-CSV users need SciPy and Matplotlib but can
import and run the pipeline without installing the scverse stack.

## R preparation and initialization port

`benisse_preprocessing.py` ports the scientific behavior of `R/prepare.R` and
`R/initiation.R`: barcode normalization and expression-order joins; exact 10x QC flags, IGH
restriction, and maximum-UMI selection for CSV input; first-occurrence V–CDR3–J ordering and
clone sizes; the variable-expression PCA and clone aggregation for `master_dist_e`; squared
encoder distances `phi`; V/J-family support `SI`; and normalized expression Laplacian `LS`.

The MuData route does not invent missing 10x QC flags. It begins from the audited AIRR adapter's
productive-heavy selection and explicitly maps that accepted selection into the same scientific
initialization.

## Frozen-R parity

On the complete committed NSCLC example, Python and frozen R preparation agree on 1,612 cells,
1,494 nodes, exact selected-barcode/cell-clone/node ordering, exact expression and `SI`, encoder
coordinates within `5e-16`, `phi` within `2e-14`, `LS` within `2e-18`, and `master_dist_e`
within `rtol=2e-6` (the established R/scikit-learn PCA tolerance). R is invoked only by this
parity fixture and other explicitly marked oracle tests.

## Corrected-v2 expectations

All requested datasets were rerun through the complete runtime path, including preprocessing and
encoding, with no R call:

| Dataset/sample | Cells | Nodes | Corrected edges | Iterations | Python plots |
|---|---:|---:|---:|---:|---:|
| BGCV09_CV0171 | 94 | 92 | 11 | 20 | 4 |
| AP4 | 203 | 203 | 22 | 19 | 4 |
| MH9143277 | 437 | 427 | 98 | 23 | 4 |
| Committed NSCLC | 1,612 | 1,494 | 1,592 | 32 | 4 |

These reproduce the earlier corrected-core results that used R-exported initialization, showing
that removal of R preprocessing does not change the corrected graphs.

## Intentional scientific migration from v1

The production R `update_A` frees both directed coordinates for each undirected edge, supplies a
Jacobian that is not the derivative of that directed objective, ignores optimizer status, and
symmetrizes afterward. Corrected v2 uses one upper-triangle variable per edge and the
finite-difference-validated symmetric derivative.

The golden fixture records both algorithms on the same four-node case: frozen production R has
edges `(0,1)`, `(0,2)`, `(2,3)` and converges at iteration 18; corrected v2 has `(0,1)`, `(2,3)`
and converges at iteration 23. This difference is asserted in the regular suite and must not be
“fixed” back to v1 parity. The paper example similarly changes from 1,691 frozen-R edges to
1,592 corrected-v2 edges.

`edge_tolerance` is recorded in v2 parameters and provenance. Its default remains `0.0` to
preserve the validated corrected expectations; changing it requires a new scientific baseline.

## Revalidation commands

```sh
BENISSE_RUN_NATIVE_EXAMPLE=1 conda run -n benisse-scirpy022 \
  python -m pytest tests/test_native_pipeline.py \
  -k native_nsclc_end_to_end_changed_v2_expectations -v

BENISSE_RUN_NATIVE_LOCAL_DATA=1 conda run -n benisse-scirpy022 \
  python -m pytest tests/test_native_pipeline.py \
  -k native_stephenson_samples -v
```

The Stephenson object remains gitignored and must not be redistributed until its processed-data
license is resolved.
