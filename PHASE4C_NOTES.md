# Phase 4c Python→R bridge — status and open interface assumptions

Status: merged by PR #21 into `develop/v2-modernization` at `0ceb21e`, then superseded as
the default by the corrected R-free v2 pipeline. This Python→R bridge is retained as the
frozen v1/paper oracle and migration aid; it is not a v2 runtime dependency. No public package
or CLI exists yet.

Phase 4d now reuses this result schema for corrected-core validation. A narrowly scoped
production fix prevents `post_analysis.R::testCor` from selecting a zero bin size on graphs
with fewer than 50 directed edge entries. It does not change the numerical R core. Real-data
validation shows that corrected Python can prune legacy R edges; this is now an intentional,
tested v2 migration rather than silent interchangeability.

## What was delivered

- `benisse_bridge.py` (internal, pre-API):
  - `BenisseRParams` — the 7 positional R params (CLI args 5–11); defaults
    reproduce the committed `example/` run. `lambda2` scales with dataset size
    (example uses 1610 ≈ BCR-cell count), so set it per dataset.
  - `build_benisse_command` / `run_benisse_r` — assemble and run
    `Rscript Benisse.R …` as a subprocess; clear errors on missing inputs and on
    non-zero R exit (captured stderr). subprocess, not rpy2 (fragile on this
    stack); the R call is hidden behind one Python function.
  - `encode_bcr_csv` — encoder via `from AchillesEncoder import encode_bcr`
    (import-and-call, the scverse-idiomatic way, not the CLI).
  - `read_benisse_network` (+ `read_sparse_graph`, `read_clone_annotation`,
    `clone_keys`) — parse `sparse_graph.txt` (`results$A`) and
    `clone_annotation.csv` into a validated `BenisseNetworkResult`: node_ids are
    the **clone keys** `v_gene_cdr3_j_gene` from the R side (honouring the 4b
    node-identity correction), edges are the upper triangle of the symmetric `A`
    (undirected convention). Asserts A square/symmetric/dimension-aligned.
  - `run_pipeline` — end-to-end from standard Benisse CSVs: encode → run R →
    parse.
- `tests/test_benisse_bridge.py` — 9 fast tests (params/command assembly,
  subprocess error handling, synthetic parsing, and parsing the **committed
  `example/` reference outputs** = real R-oracle validation without running R),
  plus 1 slow end-to-end (encode + R core, edge-set match vs reference; runs
  under `BENISSE_RUN_SLOW_TESTS=1` with `Rscript` available).

## R contract learned (grounding the bridge)

- CLI args: `exp.csv contigs.csv encoded.csv out_dir lambda2 gamma max_iter
  lambda1 rho m stop_cutoff`.
- `Benisse.R` uses **IGH-only** refined contigs for the graph; clone key is
  `paste(v_gene, cdr3, j_gene, sep='_')`; it keeps contigs whose `cdr3` is in the
  encoded matrix's `index`.
- `sparse_graph.txt` = dense space-separated `A` (no header/rownames), symmetric,
  zero diagonal, weights in (0,1]. `clone_annotation.csv` = per-node metadata
  (`v_gene,j_gene,cdr3,clsize,graph_label`, barcode rowname) in the **same order**
  as `A` (initiation.R reindexes distances back to `meta_dedup$clone` order).
- Reference `example/` run: 1494 nodes, 1691 undirected edges, weights [0.5, 1.0].

## Superseded deferral: MuData→R-input preparation

`run_pipeline` takes the three standard Benisse CSVs. Building `exp.csv` and
`contigs.csv` **from a MuData/AIRR object** was deliberately deferred, because it
is where a wrong guess causes silent scientific error and it overlaps with
public-API decisions. Specifics discovered, for whoever implements it:

- `R/prepare.R` does `gsub('-','.', barcode)` and `intersect`s with `exp` column
  names, relying on R `read.csv`'s `make.names` to dot the `exp` header too. So a
  generated `exp.csv` can keep canonical hyphenated barcodes (R dots both sides);
  do not pre-dot only one side. `assert_reversible_barcodes` still applies.
- `exp.csv` = genes × cells, gene ids in column 1 (unnamed header → R reads it as
  `X`). Source: `mdata.mod["gex"]` (decide raw layer vs `X`).
- `contigs.csv` = 10x `all_contig` format. `prepare.R` filters
  `is_cell=='True' & high_confidence=='True' & full_length=='True' &
  productive=='True' & chain=='IGH'` (string `'True'`), then per barcode keeps the
  max-`umis` row. AIRR objects do **not** carry `is_cell/high_confidence/
  full_length` as such; mapping scirpy QC → these flags is an explicit decision,
  not a mechanical rename. Emit all productive IGH chains and let R dedup, or
  reproduce R's max-umi pick in Python.
- Once implemented, wire `airr_adapter.attach_network_result(mdata, result)` so
  the MuData-native call returns the network in `uns`.

These questions are resolved for v2 by `benisse_preprocessing.py` and
`benisse_pipeline.run_mudata_pipeline`: the audited AIRR productive-heavy selection is the
accepted input contract, results attach to `uns`, and the runtime does not generate R inputs.

## Coordination with Phase 4d (Codex, in parallel)

- `read_benisse_network` is **shared infrastructure**: the 4d Python port should
  reuse it to parse the R oracle and compare against its own output. The
  strengthened parity gate (discrete edge-set / Jaccard on `sparse_graph`, per
  UPDATE_PLAN.md 4d) can be expressed directly as set equality on the
  `BenisseNetworkResult` edge triples, exactly as the slow test does here.
- The public function boundary (`run_pipeline` / `run_benisse_r` →
  `BenisseNetworkResult`) is the seam 4d slots into: 4d swaps the R subprocess
  for ported Python behind the same signature and schema.
- 4d must target the **corrected** convergence oracle (`util.R:38`,
  `sum(delta^2)/n^2`) — the `example/` reference outputs used here reflect the
  current committed `Benisse.R`; re-baseline if the convergence fix lands.

## Open interface assumptions

1. **Encoder seam.** `encode_bcr_csv` imports `encode_bcr(input, output,
   cuda=…)`. Phase 4a (Codex) is refactoring the encoder; if that signature
   changes, this one call site needs updating (4a tests pin the behaviour).
2. **Params identity.** `lambda2` is dataset-scale-dependent; there is no safe
   universal default. Consider deriving a default from node count at the
   MuData-native layer.
3. **Symmetry tolerance.** `read_benisse_network` asserts `A` symmetric within
   `atol=1e-8`. If a future core emits an asymmetric/directed graph, pass
   `directed=True` semantics through instead of upper-triangularising.
4. **Output plots.** PDF outputs are returned as paths but not parsed; plot
   parity is out of scope until the Phase 2 Python plotting rebuild.
