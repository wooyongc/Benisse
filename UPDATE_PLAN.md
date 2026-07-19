# Benisse Update Plan

Status: LIVING PLAN — corrected R-free Python v2 pipeline active; packaging still deferred
Date: 2026-07-19

Benisse is a two-stage BCR analysis tool: a Python/torch encoder embeds BCR CDR3H
sequences, then an R sparse-graph model (`Benisse.R`) relates those embeddings to
single-cell gene expression. Research code from a 2022 Nature Machine Intelligence
paper, maintained as a legacy repo.

> Context docs (local-only): `benisse_context.md` is the math/model authority (paper +
> Supplementary Note 1, equation→code map — cite by §/Eq). `mvtcr_context.md` and
> `bigcn_context.md` distill the two most relevant related methods; cite them where a plan
> item touches positioning, packaging, or the sequence encoder. This plan is kept consistent
> with all three.

## Living status and release workflow

This file is the project execution log as well as the roadmap. Update the dashboard and work
log in the same branch as each material change. Status vocabulary: **DONE** (implemented and
verified), **ACTIVE** (current branch), **PENDING** (ready but not started), **DEFERRED**
(intentionally sequenced later), and **BLOCKED** (requires a decision or external input).

### Branch model

- `main` remains the stable v1 line. Do not merge modernization work into it piecemeal.
- `develop/v2-modernization` is the long-lived v2 integration branch.
- Use public-facing topic branches such as `fix/convergence-reproducibility` and
  `feature/anndata-io`; verify each topic before merging it into the integration branch.
- Merge `develop/v2-modernization` into `main` only when the v2 release gates are complete,
  then tag the release as v2.

### Dashboard

| Workstream | Status | Evidence / next gate |
|---|---|---|
| Phase 1 audit fixes and hygiene | **DONE** | Corrected pipeline converges at iteration 33; numerical/text outputs and rendered plots match the committed oracle. |
| Reference-output baseline | **DONE** | `example/reference-output-hashes.sha256`; encoder and five stable R outputs reproduce byte-for-byte. RData is compared semantically and PDFs by rendered pixels because serialization metadata varies. |
| pandas 2 compatibility | **DONE** | Single- and multi-file encodes pass under pandas 2.3.3 with identical encoder SHA-256. |
| AIRR/scirpy research fixture | **DONE** | `data/manifest.yaml`, ignored 5k `.h5mu`, deterministic AP4 203-cell fixture, and `environment-scirpy022.yml`. |
| AIRR processed-data license | **BLOCKED** | Confirm redistribution terms before publishing either downloaded object; objects remain gitignored. |
| Large-file/history cleanup | **PENDING** | Coordinate the remaining asset migration and one repository-wide history rewrite later; document Zenodo/DOI implications first. |
| In-house cohort data | **DONE in active tree; history pending** | Archived on the existing Figshare record and removed from the repository tree at the maintainer's direction; remove its old blob during the coordinated history rewrite. |
| Phase 4a parity harness/internal modularization | **DONE; audit merged** | PR #18 added direct convergence cancellation coverage, true multi-file/order metamorphic cases, encoder schema checks, hash-ledger completeness, and R scientific invariants. |
| Phase 4b AnnData/AIRR I/O | **DONE; audit hardened** | PR #17 added the contract and PR #19 merged its fresh-context hardening into `develop/v2-modernization` at `ac52d61`. Track Awkward's experimental support and MuData 0.4's upcoming `.update()` behavior before lifting versions. |
| Phase 4c Python→R bridge | **DONE; FROZEN V1 ORACLE** | PR #21's bridge remains for paper-output reproduction and migration tests, but is no longer a v2 runtime dependency. |
| Phase 4d corrected Python core | **DONE; V2 DEFAULT** | The finite-difference-validated symmetric-edge algorithm is the intentional v2 behavior. A dual oracle freezes the four-node v1 graph (3 edges) separately from corrected v2 (2 edges). |
| R-free preprocessing/pipeline | **DONE; AWAITING MERGE** | Python ports `prepare.R`/`initiation.R`, runs corrected ADMM, returns/attaches `BenisseNetworkResult`, and generates plots. NSCLC and three Stephenson samples reproduce corrected expectations without R; the final combined suite passes (`95 passed, 8 skipped`). |
| Phase 2 Python plots | **DONE; MERGED** | PR #22 merged at `a287ad2`; the plotting layer is now called directly by the R-free pipeline. |
| Phase 4e packaging/CLI/release hardening | **DEFERRED** | Package only after the scientific API and data model stabilize. |
| Phase 3 tutorial | **DEFERRED** | Write against the final 4e CLI rather than documenting transitional entry points. |

### Work log

- **2026-07-19 — corrected R-free v2 pipeline (`feat/python-native-pipeline`).** Ported the
  scientific behavior of `R/prepare.R` and `R/initiation.R` into
  `benisse_preprocessing.py`, including QC/order joins, clone identity and size, expression PCA
  aggregation, `master_dist_e`, `phi`, `SI`, initial `A`, and `LS`. Added
  `benisse_pipeline.py` as the internal v2 default: standard CSV and MuData/AIRR workflows call
  the existing Python encoder, corrected ADMM, shared result schema, compatible output writer,
  and all four Python plots without importing or invoking the R bridge. MuData receives its
  cell embedding in `obsm["X_benisse"]` and network in `uns["benisse"]`. Complete NSCLC
  preparation matches the frozen R oracle in cell/node order and scientific matrices (exact
  expression/`SI`, `phi` within `2e-14`, `LS` within `2e-18`, `master_dist_e` within the
  established `rtol=2e-6`). Added a dual golden oracle that explicitly asserts frozen v1's
  three-edge toy graph versus corrected v2's two-edge graph. R-free end-to-end revalidation:
  BGCV09_CV0171 94 cells/92 nodes/11 edges/20 iterations; AP4 203/203/22/19;
  MH9143277 437/427/98/23; NSCLC 1,612/1,494/1,592/32. Every run produced four Python plots;
  outputs record `runtime_requires_r=false`. Migration rationale and commands are in
  `PHASE4_NATIVE_NOTES.md`. Final combined verification: `git diff --check`, Python compilation,
  and the complete test suite all passed (`95 passed, 8 skipped, 43 known dependency warnings in
  56.53s`). Packaging remains deferred until this scientific layer is reviewed and merged.

- **2026-07-18 — Phase 2 Python post-analysis and plots
  (`feat/python-post-analysis-plots`).** Added an internal implementation-neutral plotting layer
  over `BenisseNetworkResult`, retaining the Phase 4c R bridge as the default core. It aligns
  clone annotation, latent distances, and encoder embeddings; reconstructs V/J candidate support
  and true connected components; and ports the clone-expression distance and `testCor`
  aggregation algorithms. Added PCA, optional deterministic UMAP/t-SNE network layouts; clone
  size, V/J, and component color modes; latent-distance relationship boxes; clone/component and
  retained-edge diagnostics; and expression–BCR coupling scatter/statistics. Python PDFs use
  separate names and never replace the R oracle. A live R fixture confirms the complete
  1,494-node NSCLC expression-distance matrix within `rtol=2e-6`, exact candidate support, and
  both Spearman results within `2e-8`. The audit captured two legacy semantics: within-clone
  aggregation can make `master_dist_e`'s diagonal nonzero, and the final correlation-bin
  remainder is appended to the last full group. Focused Phase 2 result: 11 passed in 32.94s;
  combined fast result: 89 passed, 4 explicit local/slow skips in 41.39s. Generated full-example
  PDFs were raster-inspected successfully. Details and usage are in `PHASE2_NOTES.md`.

- **2026-07-18 — Phase 4d guarded real-data hardening after Phase 4c merge.** Merged the
  Phase 4c bridge (PR #21, `0ceb21e`) into `feat/python-r-core-port` and reused its shared
  `BenisseNetworkResult` schema for comparisons. Added internal, opt-in validation that selects
  one complete MuData sample, refuses more than 500 clone nodes by default, exports exact
  production-R initialization matrices, and requires convergence, successful inner solves,
  finite symmetric matrices, valid support, and positive-definite `Q`. A 94-cell validation
  exposed a pre-existing reporting crash when `testCor` computed a zero group size; clamped it
  to one and added a direct R regression test. Complete Stephenson samples BGCV09_CV0171,
  AP4, and MH9143277 (92, 203, and 427 nodes) all converged in 0.09–1.62 s in the corrected
  Python core. The committed NSCLC example (1,494 nodes) converged in 73.04 s. Corrected-vs-
  legacy edge Jaccard was 1.000, 0.667, 0.933, and 0.941 respectively; latent-distance
  Spearman was 0.9933–0.9985. The corrected core added no legacy-absent edges and pruned
  0, 11, 7, and 99. This is strong local stability evidence but not silent scientific
  equivalence: Phase 4c stays default pending a maintainer decision and migration notes.
  The 4,833-clone multi-patient 5k object was deliberately not treated as one graph. Detailed
  protocol and results are recorded in `PHASE4D_NOTES.md`. Verification after merging 4c:
  78 passed, 4 explicit local/slow checks skipped in 14.77s; the focused 4c/4d/R-invariant
  selection passed 42 with 3 explicit skips in 10.34s.

- **2026-07-18 — Phase 4d corrected small-fixture scope (`feat/python-r-core-port`).** A
  fresh-context audit found that the first directed-coordinate translation supplied an
  inconsistent Jacobian to SciPy, silently accepted `ABNORMAL` L-BFGS-B exits, and therefore
  could report ADMM convergence after failed inner solves. Replaced it with one variable per
  undirected upper-triangle edge and the paper's mathematically corresponding symmetric-edge
  gradient. `A` is now symmetric by construction, objective and Jacobian share one dense
  evaluation, solver metadata is retained, and any failed inner solve aborts ADMM. Replaced
  the legacy optimizer fixture with a corrected test-only R oracle and added central finite
  differences, successful-solver/KKT checks, exact small corrected-R comparisons, a complete
  four-node ADMM run, permutation equivariance, an eight-node alternate case, empty-graph and
  invalid-input cases, and allocation-free verification of the `n > 1000` policy. Production
  `R/update.R`, committed example outputs, and hashes are unchanged. The earlier 1,494-node
  diagnostics are retained only as audit history and are **not** evidence for the corrected
  algorithm. Full-example/cohort-scale scientific equivalence is deferred due to available
  compute and data bandwidth; the Python module remains internal/experimental, Phase 4c is the
  supported execution path, and R retirement is not authorized. Verification: corrected
  fixture regeneration is byte-identical; Phase 4d focused suite 19 passed in 0.63s; combined
  fast suite 57 passed, 1 legacy slow test skipped in 12.57s.
- **2026-07-18 — Phase 4b audit/hardening merged.** PR #18 merged the independent Phase 4a
  parity audit (`7fe8f7c`), and PR #19 merged Claude Code's AIRR adapter hardening (`44bff80`)
  into `develop/v2-modernization` at `ac52d61`. Phase 4c and the isolated Phase 4d numerical
  port can therefore proceed concurrently from the same audited base.

- **2026-07-18 — independent Phase 4a test audit follow-up
  (`test/scientific-parity-harness`).** A fresh-context agent found that the single full oracle
  was strong but could miss regression of the corrected convergence formula and lacked
  metamorphic/invariant checks. Extracted `graphChangeMSE` for a direct opposing-change
  cancellation test; replaced the duplicate-only multi-file assumption with disjoint file and
  order-permutation cases; asserted the callable encoder's 20-dimensional finite schema;
  asserted hash-ledger completeness; and added fast R checks for symmetry, bounds, positive
  definiteness, graph support, and latent-distance properties. Added a 30-minute subprocess
  timeout to the full harness. The audit also established that `sparse_graph.txt` is weighted
  `A`, while binary adjacency lives in `results$sparse_graph`; downstream interfaces must keep
  that distinction explicit. Fast result: 9 passed, 1 slow skipped. Full oracle after the R
  helper extraction: 1 passed in 10m17s, converging at iteration 33 with exact stable outputs,
  semantic RData equality, 3,382 binary sparse-matrix entries, and pixel-identical plots. After
  rebasing onto merged Phase 4b, the combined fast suite passed 29 tests with 1 slow skip. Its
  19 upstream warnings are retained as compatibility signals: Awkward-in-AnnData support is
  experimental and MuData 0.4 will change `.update()` pull behavior.
- **2026-07-18 — Claude Code Phase 4b merged.** Commit `61c91fb` (`Add Phase 4b AIRR/scverse
  contract adapter and tests`) was merged by PR #17 into `develop/v2-modernization` at
  `2e7a940`. It contains `airr_adapter.py`, `derive_ap4_encoder_input.py`,
  `PHASE4B_NOTES.md`, and two adapter test modules. The contract is implemented and tested;
  its documented open assumptions (notably R node ordering, result wiring, input surface,
  count precedence, and final module placement) carry into Phases 4c/4e. Phase 4a audit
  follow-up deliberately avoids modifying those files.
- **2026-07-18 — Phase 4a scientific parity harness
  (`test/scientific-parity-harness`).** Extracted a callable `encode_bcr` boundary while
  preserving the legacy CLI. Added pytest coverage for CLI boolean parsing, exact callable
  encoder output, duplicate multi-file input, and the committed reference-hash ledger. Added
  an explicitly enabled slow pytest that runs both pipeline stages in a temporary directory,
  byte-compares stable outputs, checks semantic RData and exact sparse-edge agreement, and
  raster-compares PDFs. Fast result: 4 passed, 1 slow skipped. Full oracle: 1 passed in
  11m32s; iteration 33, exact sparse matrix entries = 3,382.
- **2026-07-18 — in-house cohort archival.** The maintainer confirmed that
  `data/in-house_cohort_BCR_data.csv` was a reviewer-requested publication artifact and moved
  it to the existing Benisse Figshare record. Removed it from the active repository tree
  without adding a separate manifest. Its historical blob remains until the coordinated
  repository-wide history rewrite.
- **2026-07-18 — v2 sequencing and data-policy decisions.** Deferred distribution packaging
  and public CLI stabilization until after the AnnData work, R bridge, Python core port, and
  plots. Early Python work is limited to parity tests and the internal modular boundaries
  required by those scientific tasks. Large migrated assets will be published to both Figshare
  and a GitHub Release and removed from existing Git history; the Zenodo/DOI impact must be
  documented before rewriting. The initial decision to retain the in-house cohort was
  superseded later the same day after its Figshare archival was confirmed; it is now absent
  from the active tree and awaits historical removal.
- **2026-07-18 — Phase 1 completion (`fix/convergence-reproducibility`).** Fixed the
  convergence norm in `R/util.R`; the corrected full pipeline still converged at iteration
  33 with an identical sparse edge set and numerical results. Added a checked-in SHA-256
  oracle ledger. Lifted the pandas documentation pin after replacing all live
  `DataFrame.append` calls and verifying both single- and multi-file encoding. Pinned the
  combined Scirpy/encoder compatibility environment to Python 3.10, Scirpy 0.22.3,
  NumPy 1.26.4, pandas 2.3.3, Torch 2.2.2, Numba 0.61.2, and llvmlite 0.44.0.
- **2026-07-18 — AIRR groundwork.** Saved local AIRR/scirpy best-practice context in
  gitignored `airr_context.md`; downloaded Scirpy's Stephenson 5k MuData object into
  gitignored `data/external/`; derived a deterministic 203-cell, BCR-complete AP4 fixture;
  recorded provenance, structure, versions, and hashes in `data/manifest.yaml`. Redistribution
  remains blocked on processed-dataset license verification.
- **Previously merged — reproducibility/audit fixes (`a218ced`, `33df5f5`).** Added seeded,
  single-threaded deterministic encoding; fixed CPU checkpoint loading and `convertCluster`;
  removed dead CLI arguments and tracked repository cruft; refreshed the original example
  references and README.

## Decisions locked in
- Scope: **all four phases**.
- Large committed data files: **move to figshare/GitHub release**; keep only small toy
  inputs in-tree — BUT do not move the reference outputs a phase still verifies against
  (see Phase 1 step 0).
- Packaging target: **eventually converge on Python + AnnData, single `pip install benisse`**,
  but defer distribution packaging and public API/CLI stabilization until the scientific core
  and data model have stabilized;
  R core ported to Python test-first against the R implementation as reference oracle.
  Seurat users served via `.h5ad` export bridge, not a parallel R package.
- **Competitive positioning & differentiation.** Benisse's moat is the **interpretable convex
  sparse-graph "BCR networks"** (the support of `A` is a strict subset of the V/J crude graph —
  `benisse_context.md` §5) — an explainable, sparse output that newer methods give up. We do
  **NOT** chase BiGCN's GCN fusion (`bigcn_context.md`) or mvTCR's generative VAE
  (`mvtcr_context.md`); those are more scalable but black-box and answer different questions
  (cell-level embeddings, not sparse interpretable networks). Lead the pitch with interpretability
  + usability. Caveat: BiGCN (Small Methods 2026) **reuses Benisse's Atchley→contrastive 20-d
  encoder and V/J crude graph wholesale** — that validates the encoder but also **commoditizes**
  it, so the encoder is table-stakes, NOT the headline differentiator. The sparse-graph model is.

---

## Audit findings (grounded in current code; verified by a second independent pass)

### Real bugs / latent defects
- **DONE 2026-07-18 — `R/util.R:38` convergence-criterion operator-precedence bug.**
  `sum(res[[r]]-res_back[[r]])^2/nrow(sparse_graph)^2` parses as `(sum(Δ))^2`, i.e. the
  **square of the sum**, not `sum(Δ^2)` (squared Frobenius norm) as intended. Since
  `sparse_graph` is 0/1, elementwise deltas are −1/0/+1 and cancel in the sum, so the
  stop metric under-measures change and can trigger early/false convergence. Genuine
  correctness bug in the R core AND a landmine for the Phase 4d port (a "fixed" port will
  diverge from the wrong oracle). Confirmed against documented intent: `benisse_context.md`
  §8 states the stop criterion is the "mean squared per-entry change" (= `sum(Δ²)/n²`), which
  is exactly what the code fails to compute.
- **DONE (`a218ced`)** — `post_analysis.R:74/:81` — `convertCluster(sparse_graph)` ignored its own parameter and
  reads the **global** `results$sparse_graph`. Works only because `Benisse.R:92/:97` has a
  global `results` passed as the identical object. Genuine latent bug.
- **DONE 2026-07-18** — pandas 2 compatibility. The encoder already accumulated batches in
  a list; both remaining loader `DataFrame.append` calls were replaced with `pd.concat` and
  single-/multi-file parity was verified under pandas 2.3.3.
- **DONE (`a218ced`)** — `AchillesEncoder.py` CPU/CUDA checkpoint loading. The root cause was:
  `:99 if opt.cuda:` branches on the **flag**, not the resolved `device` from `:85`. On a
  CPU-only machine `--cuda True` sets `device=cpu` but `:100 torch.load(...)` still runs
  without `map_location` and errors if the checkpoint holds CUDA tensors. README `:111`
  ships `--cuda True` as the example → the failing invocation. Fix: always pass
  `map_location=device`, branch on `device`.
- **DONE (`a218ced`)** — `--atchley_factors` / `--model` were declared but never used;
  paths hardcoded (`:57`, `:100/:102`). Dead/misleading.

### Style / robustness
- `prepare.R:7` uses `attach(contigs)`/`detach()` — deprecated, fragile scoping.
- `getLatentTdist` recomputed >=4x per run (in `checkDist`, in `testCor`
  `post_analysis.R:97` via `plotClusters`, and `Benisse.R:103`).
- `CMC/contrast_util.py:109` `.cuda()` and `CMC/alias_multinomial.py:46-47` — training path
  is not CPU-runnable (inference/encode path never reaches these, so encoding is safe).
- `CMC/model_util.py:9` `nn.DataParallel` wrap — adds `module.` state-dict prefix, pointless
  on CPU; a clean package should unwrap it.
- `AchillesEncoder.py:44` `--encode_dim` help says "default: 80" but `default=40`;
  `--pad_length` is user-settable yet `in_feature=130` is hardcoded (`:88`), so changing
  `--pad_length` silently breaks the model.
- Hardcoded `batch_size`/seeds/hyperparams (`AchillesEncoder.py:65-66,:87-95`); R
  hyperparams read positionally (`Benisse.R:73-79`).

### Repo hygiene / docs / data governance
- **DONE (`a218ced`)** — removed committed junk: `.Rhistory`, `.idea/.gitignore`, `CMC/.DS_Store`,
  `R/.DS_Store`, `example/.DS_Store`, `figs/.DS_Store`.
- **DONE (`a218ced`, Phase 1 completion)** — extended `.gitignore` for OS/editor cruft,
  local agent/context files, downloaded AIRR objects, and regenerable graph output.
- Repo bloat (git blobs): `example/Benisse_results.RData` ~49M, `latent_dist.txt` ~36M,
  `10x_NSCLC_exp.csv` ~15M, `data/in-house_cohort_BCR_data.csv` ~10M,
  `example/sparse_graph.txt` ~4.3M, `example/cleaned_exp.txt` ~3.1M, `figs/*.png` ~3.2M.
- **`data/in-house_cohort_BCR_data.csv` (NEW governance item).** Unreferenced 10 MB
  human-subjects BCR data (committed in `60161c4`), no consent/governance note. `git rm`
  won't remove it from history — needs a data-governance decision + history scrub.
- **DONE (`a218ced`, Phase 1 completion)** — README defects: `pip install sklearn` (broken stub); Python 3.7 / R 4.0.2;
  `:20` torch 1.10 / pandas 1.3.4 / sklearn 1.0 / numpy 1.21.3; `:67/:69` broken hyperlink
  filenames (`_contig.csv` vs `_contigs.csv`, `_contig_exp.csv` vs `_exp.csv` — display
  text is correct); `:30-59` venv instructions conflicting with the conda workflow.
- No test suite, no CI, no packaging metadata.

---

## Phase 1 — Audit fixes & hygiene — DONE 2026-07-18
**Step 0: snapshot content hashes** of all current reference outputs
(`encoded_10x_NSCLC.csv`, `sparse_graph.txt`, `latent_dist.txt`, `clone_annotation.csv`,
etc.) and record seed determinism. **DONE:** the committed baseline is recorded in
`example/reference-output-hashes.sha256`. Stable text outputs are byte-compared; `.RData`
is compared object-by-object and PDFs by rasterized pages because their metadata is volatile.

1. **DONE:** `convertCluster` uses its parameter, not the global.
2. **DONE:** fixed `util.R:38` to compute `sum(delta^2)/n^2`. The corrected oracle still
   converges at iteration 33 and produces the identical edge set, so no scientific rebaseline
   was needed.
3. **DONE:** encoder always loads the checkpoint with `map_location=device`.
4. **DONE:** removed dead `--atchley_factors` / `--model` arguments.
5. **DONE:** extended `.gitignore` and removed tracked OS/editor/Python cruft.
6. **DONE:** fixed README dependencies, filenames, versions, and conda workflow.
7. **DONE:** reran both pipeline stages with `--cuda False` and verified against the oracle.

### Split out of Phase 1 (own PRs)
- **DONE on the Phase 1 completion branch — pandas-pin lift.** Replaced the two remaining
  loader `.append` sites, lifted the documented pin to pandas 2.3.3, and verified exact
  single-/multi-file encoder parity. Kept as a focused change within the public-facing
  reproducibility branch because it was already independently audited before the R fix.
- **DECIDED; implementation pending — large-file move + history rewrite.** Publish migrated
  repository assets to both Figshare and a GitHub Release, then use `git filter-repo`/BFG for
  real bloat reduction. HIGH blast radius: document the Zenodo DOI archive implications
  (README badge line 1) before rewriting every historical commit hash. This decision does not
  authorize redistribution of the separately downloaded AIRR fixture.
- **DONE in active tree; history cleanup pending — `data/in-house_cohort_BCR_data.csv`.** The
  maintainer confirmed it was shared on Figshare as a reviewer-requested publication artifact
  and directed its removal from Git. Do not restore it or create a separate repository
  manifest; include the old blob in the later coordinated history rewrite.

## Phase 2 — New plotting functions
**DONE on `feat/python-post-analysis-plots`; PR pending.** The implementation-neutral layer consumes
`BenisseNetworkResult`, annotation, latent distances, cleaned expression, clonality labels, and
the encoder embedding rather than depending on an RData object. Delivered scope includes:
clonotype networks colored by clone size / V-J family / true graph component; PCA plus optional
UMAP/t-SNE coordinates with graph edges; clone-size, component-size, and edges-remaining
diagnostics; the latent-distance relationship groups from `checkDist`; and expression-vs-latent
and expression-vs-encoder coupling scatter/statistics from `testCor`. The Python reconstruction
of `master_dist_e`, V/J candidate support, and both coupling correlations is checked directly
against the committed R oracle. Generated PDFs use distinct `python_` names. See
`PHASE2_NOTES.md`; public plotting API/CLI choices remain deferred to 4e.

## Phase 3 — User-friendliness & tutorials
- Rewrite README against actual filenames and the conda/CPU workflow.
- End-to-end runnable tutorial (notebook) on the toy dataset.
- NOTE: defer the final wrapper, named flags, and tutorial until 4e so they document the
  stable Python core and AnnData data model rather than transitional interfaces.

## Phase 4 — Scientific modernization first; packaging last
Rationale: encoder already Python/torch; contig input is standard 10x `all_contig` format
(`prepare.R:8-10`) which scirpy reads natively; scanpy/scirpy is the immune-repertoire
ecosystem. R core is a self-contained numerical routine; `update_Q` (eigendecomposition
matrix function, `update.R:51-60`) ports parity-safe, but `update_A` (bounded L-BFGS-B,
`update.R:1-49`) is the risk (R `factr`/`maxit` vs scipy `ftol`/`gtol`/`maxiter`; path
dependence compounds across ADMM iterations).

**Competitive stakes for this phase:** AnnData-native is not just convenience — it's a live
gap to exploit. BiGCN ships `.xlsx/.csv/.pt` on Python 3.7 and is **not** AnnData-native
(`bigcn_context.md`), so a clean AnnData/scirpy Benisse is strictly more interoperable *today*
(treat as a near-term opportunity, not an evergreen moat — competitors close it fast). mvTCR is
a **working template** for exactly this design (AnnData/scirpy-native, `adata.obsm` I/O, PyPI
`mvtcr`, scArches reference mapping — `mvtcr_context.md`); study it as prior art for 4b/4e rather
than designing the packaging from scratch.

- **DONE — 4a scientific parity harness + minimal internal modularization.** Added automated checks for
  exact encoder output, stable R text outputs, semantic RData equality, rendered-plot parity,
  and exact sparse edge-set agreement. Extracted only the callable internal encoder boundaries
  needed by tests and later adapters. Do **not** publish a package, promise a public Python API,
  redesign the CLI, or rearrange model assets yet.
- **DONE — 4b AnnData/AIRR I/O:** read/write `.h5ad`/`.h5mu`; embeddings into `adata.obsm`.
  AIRR best practices are distilled in local `airr_context.md`; the
  ignored Stephenson 5k MuData reference and deterministic AP4 203-cell BCR fixture are
  recorded in `data/manifest.yaml` with checksums and provenance; Scirpy 0.22.3 is working
  on this Intel Mac via `environment-scirpy022.yml`. The merged adapter defines deterministic
  heavy-chain selection and field mapping, includes a reproducible fixture derivation script,
  and tests embedding writes without destroying the AIRR modality. Do not publish the
  downloaded/derived objects until their processed-data redistribution license is confirmed.
- **DONE — 4c interim Python→R bridge; frozen v1 oracle.** The thin subprocess wrapper remains
  available for published-output reproduction and migration tests, not v2 runtime execution.
- **DONE — 4d corrected Python core; v2 default.** The Python core uses the mathematically
  correct symmetric-edge parameterization and is tested against finite differences, corrected R,
  three complete Stephenson samples, and the 1,494-node paper example. A separate production-R
  fixture freezes the known topology gap. The corrected graph's edge pruning is intentional v2
  behavior and is documented rather than hidden behind a parity claim.
  **Why port ADMM rather than replace it:** the ADMM sparse-graph is the differentiator —
  convex, interpretable, and it guarantees the learned graph is a strict sparse subset of the
  crude graph (`benisse_context.md` §5). BiGCN's GCN fusion (`bigcn_context.md`) is a known
  successor path that gives exactly this up, so we deliberately keep ADMM. We do NOT hedge the
  port — leaning in is the strategy — but we validate the choice empirically via the competitive
  benchmark below, not by abandoning it.
- **DONE — Phase 2:** Python plots consume the shared result object and are wired into the
  corrected R-free runtime.
- 4e — Package and release-harden last. Add `pyproject.toml`, the installable `benisse/`
  package, console entry point, model assets as package data, and the final named CLI. Strip
  the runtime DataParallel wrapper (`model_util.py:9`) while retaining checkpoint compatibility;
  document the training-path `.cuda()` limitation; add CI, installation tests, migration notes,
  the tutorial, and the Seurat bridge (`sceasy`/`zellkonverter`). Retire/legacy the required R
  stage only after the Python parity gate passes.

Start the test suite in 4a; defer distribution/install/CLI CI to 4e.

---

## v2 (out of current scope) — Atchley → ESM2 / protein-LM embedding upgrade

Assessment of replacing the hand-crafted Atchley featurization with a pretrained protein
language model (ESM2 or an antibody-specific PLM). Grounded in the actual encoder.

### What the current encoder actually is (inference path)
- Two-view CMC (Contrastive Multiview Coding). View 1: CDR3H aa seq → Atchley factors
  (5-dim/residue, `data_pre.py:63-77`) padded to `encode_dim=40` → 2D CNN
  `alexnet_cdr_to_vdj` (`model_util.py:30-76`) → `feat_dim=20`. View 2: nucleotide seq →
  ordinal encoding → 1D CNN.
- KEY: at inference the nt/VDJ view is **mocked** (`data_pre.py:56` fixed dummy nt) and its
  output is **discarded** — only `feat_cdr` is written (`AchillesEncoder.py:122-124`). So the
  R model consumes a 20-dim vector from the frozen Atchley→CNN branch alone. The contrastive
  training only produced the frozen weights; it is not exercised at inference.
- Consequence: swapping to ESM2 = replacing one frozen featurizer/encoder with another. It
  does NOT require reproducing the contrastive setup at inference time.

### Would it work? Yes. Two strategies, very different effort.

**Strategy A — raw PLM embedding as a drop-in encoder (recommended first step).**
CDR3H → ESM2 → mean-pool over residues → (optional linear/PCA projection) → feed R model.
- No retraining of a contrastive model needed. ESM2 embeddings used directly.
- Dimension handling is the main code touch: pooled ESM2 is 320-1280 dim; the R stage
  hardcodes a 20-col slice (`Benisse.R:68` `contigs_encoded[...,1:20]`). Either project ESM2
  → 20-dim (PCA/linear) or generalize that slice + `m`. Modest, localized change.
- Compute on this Intel Mac (CPU, no CUDA): esm2_t6_8M (320-dim) / t12_35M (480-dim) run fine
  on CPU for short CDR3H (~10-25 aa) — seconds-to-minutes on the toy set. 650M/3B are
  CPU-slow but toy-feasible; large cohorts want GPU. Dependency: `fair-esm` or HF
  `transformers`; weight download ~30MB (8M) to ~2.5GB (650M).
- Effort: ~1-2 weeks. Prototype in days. Bulk of the work is revalidation — this changes the
  scientific output, so benchmark ESM2-vs-Atchley embeddings through the full pipeline on the
  paper/toy data (graph quality, expression-latent correlation) before adopting.

**Strategy B — keep the CMC framework, swap Atchley for per-residue ESM2 features, retrain.**
Feed ESM2 (L×320) into the CNN instead of Atchley (L×5); retrain `trained_model.pt`.
- Requires: the original training corpus (large BCR repertoire — NOT in repo; referenced only
  by dead server pkl paths at `data_pre.py:110-115`), the training script
  (`CMC/AchillesEncoder_train.py` exists), and GPU (contrastive NCE training over large N is
  GPU-bound; `contrast_util.py:109` hardcodes `.cuda()` — training is not CPU-runnable).
- Effort: HIGH, research-grade — ~1-3 months + GPU + data access + revalidation. Only justified
  if Strategy A shows the downstream model benefits from richer embeddings.

### Better target than vanilla ESM2?
ESM2 is trained on general UniRef proteins. For BCR/antibody CDR3H specifically, antibody-
specialized PLMs often embed better: AntiBERTy, AbLang/AbLang2, IgBert, BALM, or ESM fine-
tuned on OAS. Caveat: CDR3H in isolation (no framework/VH context) is short and hypervariable;
some antibody PLMs expect fuller VH. If investing, evaluate an antibody-specific PLM alongside
ESM2 rather than assuming generic ESM2 is the ceiling.

Related-method evidence (`mvtcr_context.md`): mvTCR encodes CDR3 with **transformers trained
from scratch** (not a pretrained PLM) and still beats hand-crafted/dataset-level baselines. Two
takeaways: (a) independent support that a **learned** sequence encoder beats a hand-crafted
featurization for CDR3 — strengthens the case for moving off Atchley; (b) but it is **agnostic
on pretrained-vs-scratch**, so it does NOT specifically vindicate ESM2. ESM2's edge remains
"skip training entirely," which is exactly why Strategy A is ranked first — no change to the
A-first ordering.

### v2 recommendation
1. Do Strategy A as a cheap ablation FIRST — it answers "do richer embeddings even help the
   sparse-graph model?" for ~days of work.
2. Fits the Phase 4 Python/AnnData packaging cleanly (ESM2 is Python/torch; drop the Atchley
   CSV + CNN weights in favor of a PLM call).
3. Escalate to Strategy B or antibody-PLM fine-tuning only if A is promising.
4. Always keep the Atchley encoder as the reference/baseline for benchmarking — do not delete
   `dependency/trained_model.pt` or `Atchley_factors.csv` when adding a PLM path.

---

## Phase 5 (candidate) — Differential / comparative analysis between conditions

Motivation: Benisse today is descriptive per-sample (one sparse clonotype graph + latent
distances per dataset). Most biological/clinical value in single-cell tools comes from
contrasts (responder vs non-responder, pre/post therapy, tumor vs normal, vaccine timepoints).
Adding a principled between-condition comparison layer is high value. NOTE: the R core has a
thin `sample` hook (`R/initiation.R:7-11`, `util.R` `Benisse(..., sample=NA)`) but it only
prefixes V/J crude-cluster labels for multi-sample-in-one-library runs — NOT a differential
mechanism.

### Paper precedent — the sound comparison already exists in the method
`benisse_context.md` §9 shows the authors' COVID-19 analysis (Fig. 3g / SN1 Fig. 8) is itself
a between-condition comparison: the **distribution of the BCR↔expression coupling correlation**
is tighter (higher mean, smaller SD) in severe/recovering than in cured/healthy. That coupling
is the model's core validation metric (§9): `cor(a,b)` vs `cor(a,c)` where a = B-cell RNA-
expression distances, b = latent BCR distances, c = original BCR distances (success = latent
beats original, `cor(a,b) > cor(a,c)`). It is **already computed per run** by `testCor`
(`post_analysis.R:96-120`, returns `c1=cor_ab`, `c2=cor_ac`).
Implication for Phase 5: the coupling correlation is an **invariant scalar/distribution** — it
sidesteps the node-identity and non-comparable-latent-space blind spots entirely, has published
precedent, and needs almost no new math. Make it the PRIMARY comparison axis; treat graph-
topology and node-level diffs as secondary layers built on top.

### Blind spots that make naive "diff two adjacency matrices" wrong
1. **Node-identity mismatch (central obstacle).** BCR CDR3H is hypervariable; two conditions
   (esp. two donors) share almost no identical clonotypes. Cannot subtract A vs B — node sets
   barely overlap. Need CDR3H correspondence (V-J + edit-distance clustering) OR node-set-
   agnostic graph-level/distributional comparison.
2. **Latent spaces not comparable across runs.** Each run learns its own `Q`/embedding on that
   condition's expression (absorbs batch effects). Raw `latent_dist.txt` is not comparable
   across conditions. Need a joint/anchored embedding or comparison of invariants only.
3. **Size/depth confounds.** Clonotype count, clone-size distribution, seq depth differ; graph
   density/edges/components scale with N (the `reEdge` ratio is size-dependent). Must normalize
   for node count + clone-size distribution or detect sampling artifacts.
4. **n=1 inference.** Needs replicates (multiple donors/condition) + a null (permute condition
   labels across donors, bootstrap over cells) to separate biology from within-condition noise.
5. **Run stochasticity.** The `util.R:38` convergence bug is fixed; still establish within-
   condition graph reproducibility (edge stability) before trusting between-condition diffs.

### Suggested features (soundness order)
- Alignment-free per-condition graph summaries (density, degree dist, modularity, assortativity,
  component-size dist, clustering coeff) + KS/earth-mover comparison of the within- vs cross-
  cluster latent-distance distributions. Node-set-agnostic; do first.
- Matched-clone differential: cluster CDR3H across conditions, then per matched clone compute
  differential clone size (expansion/contraction), differential connectivity ("rewiring score"),
  and differential expression of that clone's cells.
- Joint/anchored embedding (co-train with condition covariate, or integrate expression first)
  so latent distances are comparable — fixes blind spot #2. CAUTION (`mvtcr_context.md`): mvTCR
  found **naive expression integration hurts** — Harmony dropped NMI 0.456→0.212, recovered only
  by a conditional model + scArches. So treat this step as risky and keep the alignment-free
  coupling-correlation as the PRIMARY axis (as above); rank joint-embedding below it.
- **Reference mapping (NEW axis, from mvTCR).** scArches-style "map a new sample onto a fixed
  reference/atlas" (`mvtcr_context.md`) — a comparison mode Benisse has no analogue for today,
  distinct from the pairwise between-condition diffs. Lets a new dataset be positioned against an
  established Benisse reference instead of only A-vs-B. Higher effort; treat as a stretch axis.
- Differential connectivity with a null model (permutation over donor labels; bootstrap CIs).

### General (non-comparative) calculation features that raise value
- Edge stability / bootstrap edge-confidence on the sparse graph (also cures blind spot #5).
- SHM (somatic hypermutation) overlay — the affinity-maturation mechanism Benisse claims to
  capture; test whether graph neighbors share mutational lineage.
- Clonotype centrality / hub detection.
- Affinity-maturation trajectory: MST/diffusion ordering over latent distances → clonal-evolution
  pseudotime.
- Isotype / class-switch overlay (IgM/IgG/IgA from contigs) on the graph.

### Node/network-level differential via prime & trajectory groups (biologically grounded)
`benisse_context.md` §10 gives ready-made per-network readouts that form monotone gradients:
each BCR network has a **prime** clonotype (latest-created by Monocle2 pseudotime) and
**trajectory groups 1/2/3** (by latent distance from the prime; group 1 = closest). Activation-
signature score, clonal size, and class-switch rate all peak in group 1 and decline outward.
Comparative use: compare the **gradient itself** across conditions (e.g. does the group-1
activation advantage steepen in responders?) — a low-dimensional, node-alignment-free contrast
that is directly interpretable as "sharper affinity maturation." This turns the paper's own
descriptive framework into a differential readout.
Cross-check note: BiGCN (`bigcn_context.md`) targets the same B-cell functional-state and
maturation→class-switch biology; validate these Phase-5 readouts (and the class-switch overlay
above) against its published claims where they overlap.

### Visualization design (how to actually SEE the comparison)
Grounded in what Benisse emits (`sparse_graph.txt`/`A`, `latent_dist.txt`, `clone_annotation.csv`
with `graph_label`+`clsize`, `testCor` `cor_ab`/`cor_ac`, plus prime/group + activation/class-
switch from §10). A layered system, cheap→rich, matching the soundness order:

1. **Coupling-strength comparison (primary; paper precedent, alignment-free).**
   Grouped violin/box of the per-network (or bootstrap-resampled) coupling correlation, one
   group per condition — the Fig. 3g idiom. Overlay paired `cor_ab` vs `cor_ac` (points +
   connecting line) so "latent beats original" is visible within each condition. Add replicate
   donors as jittered points. Reads as: "coupling is tighter in condition X." Earth-mover/KS
   distance annotated between conditions.
2. **Graph-summary slopegraph / radar.** Alignment-free stats (density, modularity, mean degree,
   #components, edges-remaining ratio) as a slopegraph (A→B line per metric, replicate CIs) or a
   two-series radar. One glance at which structural properties move.
3. **Within- vs cross-network latent-distance distributions, faceted by condition.** Extends the
   existing `in_cross_dist_check` plot to a small-multiple (condition × distance-class); the shift
   between facets is the structural change.
4. **Differential / merged network graph (the intuitive "two graphs" picture).** Requires CDR3H
   node-matching across conditions. Lay the union graph out ONCE (shared coordinates); color
   EDGES by presence — condition-A-only / B-only / shared — and color NODES by clone-size log2FC
   (expanded vs contracted), size by mean clone size. This is the single most legible "compare
   the graphs" visual; gated on the matching step, so it comes after the alignment-free layers.
5. **BCR-network membership alluvial/Sankey.** Clonotypes flow from A's networks to B's networks,
   exposing merging/splitting of BCR networks between conditions (needs node-matching).
6. **Differential-connectivity volcano.** Per matched clonotype/cluster: x = connectivity/rewiring
   log2FC, y = −log10(permutation p), colored by significance. Familiar to biologists; needs the
   replicate/null-model layer.
7. **Trajectory-group gradient panels.** Per condition, group 1/2/3 on x, activation / clonal
   size / class-switch on y (small multiples); overlaying conditions shows whether the affinity-
   maturation gradient sharpens. Directly visualizes the §10 differential above.
8. **Prime-centered mirror ("starburst").** For a network present in both conditions, radial
   layout centered on the prime, spoke length = latent distance to prime; condition A left,
   B right, mirrored — shows contraction/expansion of a single network's structure.

Design notes: keep one color encoding for CONDITION across every panel (a two-condition
qualitative pair) and a separate diverging scale reserved for log2FC (expanded↔contracted);
never reuse the condition hue for magnitude. Everything must render in light and dark and be
colorblind-safe. When building any of these, load the `dataviz` skill before writing chart code.
Layers 1–3 need no node matching and ship first; 4–8 depend on the matching / replicate layers.

### Sequencing
Do AFTER Phase 4 (Python/AnnData core) — a fair comparison needs the joint embedding and
replicate handling that are far easier in the packaged Python stack — and it depends on the
Phase 1 convergence-reproducibility fix. Start with alignment-free summaries + edge stability
(cheap, sound, no node matching), then matched-clone differential, then joint embedding.

---

## Competitive benchmark (after 4d / alongside Phase 5)
The single missing piece a *revival* needs: an explicit head-to-head that defends the "keep
ADMM" bet and answers the reviewer/user question "why Benisse over BiGCN/mvTCR?".
- **Primary target = BiGCN.** It is the clean comparison: same domain, and it consumes the **same
  encoder input** as Benisse (Atchley→contrastive 20-d + V/J crude graph — `bigcn_context.md`).
  Benchmark Benisse's interpretable sparse networks vs BiGCN's GCN embedding on functional-state
  annotation and maturation-trajectory recovery, plus an interpretability/sparsity contrast that
  plays to Benisse's strength.
- **Gate on access:** BiGCN's fusion internals, loss, and training are **paywalled/not public**
  (`bigcn_context.md`) — obtain the full paper + supp (and run the authors' `Lxc417/BiGCN` repo)
  BEFORE making any firm benchmarking or "successor" claim. Do not plan as if the architecture
  is known.
- **Secondary:** cite mvTCR as external precedent for the comparative idioms (between-donor,
  time-course, reference mapping) rather than re-deriving them.
- Output: a positioning subsection for the revived README / release notes.

## Sequencing summary (living)
**DONE:** Phase 1 fixes + hash baseline + pandas-pin lift + AIRR/Scirpy fixture groundwork +
Phase 4a scientific parity harness and callable encoder boundary.

**DONE:** Phase 2 Python post-analysis and plots now operate on the shared result contract without
depending on which numerical core produced it.

**AWAITING MERGE:** the R-free scientific workflow and corrected-v2 reference expectations are
implemented, hardened, and revalidated; R remains only as a frozen oracle. After merge, write the
end-to-end tutorial, then proceed to 4e (package, public CLI, CI, migration notes, and Seurat bridge) → competitive
benchmark vs BiGCN (gated on paper access) → Phase 5. Merge the integration branch to `main`
and tag v2 only after the release gates below pass. Coordinate the remaining large-asset move
and one history rewrite before that release; the in-house cohort file is already absent from
the active tree and its historical blob should be removed in that rewrite.

### v2 release gates

- Fresh environment install succeeds on supported CPU platforms.
- Packaged encoder and AnnData/AIRR round-trip tests pass on the deterministic fixture.
- Python-facing end-to-end workflow reproduces corrected-v2 reference expectations; the known
  frozen-v1 topology gap is separately asserted and documented.
- Runtime and tutorial complete without R; frozen-R oracle checks remain optional validation.
- Large-file location, Zenodo/DOI implications, and human-subject data governance are resolved
  and documented.
- CI smoke tests, user tutorial, migration notes, and license/redistribution statements pass
  review.
