# Phase 4d corrected Python numerical core — validation notes

Status: **adopted as corrected v2 behavior and the default internal execution path**. The
NumPy/SciPy core is mathematically and numerically hardened and intentionally differs from the
legacy R optimizer on some graph edges. The remaining R preprocessing/initialization dependency
has now been removed; see `PHASE4_NATIVE_NOTES.md`. Public packaging is still deferred.

## Audit hardening completed

- Represent each undirected edge once, using upper-triangle optimization variables and the
  corresponding symmetric-edge gradient.
- Check the analytic Jacobian with central finite differences.
- Treat every failed inner optimizer as fatal; retain its diagnostic metadata.
- Require ADMM convergence, finite symmetric `A`, `Q`, and `R`, a zero `A` diagonal, crude-graph
  support, and positive-definite `Q` before accepting a validation result.
- Compare node ordering, graph support and weights, connected components, and latent distances
  through the shared `BenisseNetworkResult` schema introduced in Phase 4c.
- Guard local MuData validation at 500 clone nodes by default and require one biological sample,
  avoiding an accidental dense all-patient run.
- Fix `R/post_analysis.R::testCor` for graphs with fewer than 50 directed edge entries by clamping
  its bin size to one; a direct regression test covers the previously observed crash.

## Local scientific-validation results

The Stephenson examples use complete individual samples from the locally downloaded 5k MuData
object. Production R preparation and initialization export the exact `phi`, `si`, `ls`, and node
order consumed by both cores. The NSCLC case uses the committed paper example and its existing
encoded/reference outputs.

| Case | Selected cells | Graph nodes | Python core | Iterations | Legacy→corrected edges | Jaccard | Latent Spearman |
|---|---:|---:|---:|---:|---:|---:|---:|
| BGCV09_CV0171 | 94 | 92 | 0.09 s | 20 | 11→11 | 1.000 | 0.9933 |
| AP4 | 203 | 203 | 0.26 s | 19 | 33→22 | 0.667 | 0.9958 |
| MH9143277 | 437 | 427 | 1.62 s | 23 | 105→98 | 0.933 | 0.9985 |
| Committed NSCLC example | 1,612 selected cells | 1,494 | 73.04 s | 32 | 1,691→1,592 | 0.941 | 0.9981 |

Every corrected run converged, every inner optimizer succeeded, and every `Q` was positive
definite. The corrected core added no edges relative to the legacy results; it retained a subset
and pruned 0, 11, 7, and 99 edges respectively. Latent geometry is highly concordant, but AP4's
11-of-33 edge removal is scientifically material. This evidence supports stability and feasible
local execution, not silent equivalence or automatic R retirement.

The complete 5k object was not run as one graph: it combines patients and contains 4,833 clone
nodes, making a dense cross-patient optimization scientifically inappropriate as well as wasteful
on this machine.

## Reproducing local validation

Local data and generated outputs stay outside Git. Run complete samples independently:

```sh
conda run -n benisse-scirpy022 python real_data_validation.py \
  data/external/stephenson2021_5k.h5mu /tmp/benisse-bgcv09 --sample-id BGCV09_CV0171
conda run -n benisse-scirpy022 python real_data_validation.py \
  data/external/stephenson2021_5k.h5mu /tmp/benisse-ap4 --sample-id AP4
conda run -n benisse-scirpy022 python real_data_validation.py \
  data/external/stephenson2021_5k.h5mu /tmp/benisse-mh9143277 --sample-id MH9143277
```

Run the committed NSCLC corrected-core milestone explicitly with:

```sh
BENISSE_RUN_PYTHON_CORE_EXAMPLE=1 conda run -n benisse-scirpy022 \
  python -m pytest tests/test_real_data_validation.py \
  -k corrected_core_on_committed_nsclc_example -v
```

Redistribution terms for the downloaded processed Stephenson objects remain unresolved. Do not
commit the `.h5mu` objects or generated derivatives until that license question is closed.
