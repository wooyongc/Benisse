# Phase 4b AIRR/scverse contract — status and open interface assumptions

Status: independent Phase 4b contract implemented and tested against synthetic
objects and the local AP4 fixture. No public package/API created yet; this is
internal groundwork to be integrated after Phase 4a lands.

## What was delivered

- `airr_adapter.py` — internal (pre-API) adapter:
  - `select_heavy_chains` — deterministic one-heavy-chain-per-cell selection.
  - `airr_to_dataframe` — flatten scirpy's ragged `obsm["airr"]` awkward array.
  - `scirpy_primary_heavy_index` — read scirpy's `chain_indices.VDJ[0]` pointer.
  - `to_r_barcode`/`from_r_barcode`/`assert_reversible_barcodes` — explicit,
    reversible `-`<->`.` barcode join normalisation (never mutates the object).
  - `build_encoder_input`/`write_encoder_input_csv` — AIRR -> encoder columns.
  - `attach_embedding` — write embeddings into the AIRR modality obsm without
    disturbing `airr`/`chain_indices`.
  - `BenisseNetworkResult` + `validate_network_result` + `attach/read_network_result`
    — clonotype-network result schema stored in `uns`, not cell-aligned `obsp`.
  - `TEN_X_TO_AIRR`, `AIRR_TO_ENCODER` — documented field mappings.
- `tests/test_airr_adapter.py` — 14 synthetic-object tests (always run).
- `tests/test_airr_adapter_ap4.py` — 6 real-fixture parity tests (skip if the
  gitignored fixture is absent).
- `derive_ap4_encoder_input.py` — reproducible derivation of the encoder input
  CSV from the AP4 fixture, writing to gitignored `data/external/` with a
  provenance sidecar.

## Verified facts (benisse-scirpy022 env, `python -m pytest -q`: 24 passed, 1 skipped)

- Our independent heavy-chain ranking agrees with scirpy's `chain_indices`
  primary VDJ pointer on **all 203** AP4 cells.
- Derived counts match `data/manifest.yaml` `airr_summary`:
  216 productive IGH with junction_aa, 214 unique, loci IGH/IGK/IGL =
  233/138/121, 203 cells with a productive heavy chain.
- Encoder `cdr3` <- AIRR `junction_aa` (keeps conserved C..W/F residues, matching
  the committed `example/10x_NSCLC.csv`), NOT the narrower `cdr3_aa`.

## Selection contract (deterministic)

Candidate = `locus in {IGH}` AND non-empty `junction_aa` AND (`productive`).
Rank per cell by: abundance count DESC, then `junction_aa` ASC, then
`sequence_id` ASC. This is a total order, so the winner is independent of input
chain order (tested). Abundance count coalesces `umi_count` -> `duplicate_count`
-> `consensus_count` (AP4 has no `umi_count`).

## Open interface assumptions — resolve during 4a integration

1. **Adapter placement / public surface.** Currently a repo-root module imported
   directly by tests (like `AchillesEncoder`). Phase 4e must decide whether it
   moves into the `benisse/` package and what, if anything, becomes public API.
2. **Heavy loci scope.** `HEAVY_LOCI = ("IGH",)` — BCR only. If TCR support is
   ever in scope, extend to TRB/TRD and rename the "heavy" concept to "VDJ".
3. **Count-field precedence.** Assumed `umi_count > duplicate_count >
   consensus_count`. Matches scirpy on AP4, but confirm against the intended
   Scirpy 0.24.0 pin (manifest `intended_scirpy_pin`), which may expose
   `umi_count` directly and could change tie ordering.
4. **Encoder identifier (`contigs`).** Mapped from AIRR `sequence_id` (unique per
   selected chain). The legacy example used the 10x `contig_id`, which scirpy
   also stores as `sequence_id`, so this matches — but the encoder ignores this
   column at read time (`CMC/data_pre.py` reads only `contigs`,`cdr3`); it only
   matters for the barcode<->chain<->embedding join, which lives in the adapter.
5. **Barcode reversibility precondition.** `-`<->`.` normalisation is only
   reversible when barcodes contain no `.`. True for AP4; `assert_reversible_barcodes`
   guards it. A dataset with dotted barcodes needs a different join key.
6. **Result-schema node identity.** `BenisseNetworkResult.node_ids` are the
   encoder `contigs` (one per selected heavy chain). When the R core actually
   produces `sparse_graph.txt` (Phase 4c/4d), confirm its node ordering maps 1:1
   to this node index before populating `edges`. The schema is defined and
   validated but not yet wired to real R output.
7. **Embedding obsm alignment.** `attach_embedding` writes a cell-aligned matrix
   into the `airr` modality with NaN rows for cells lacking a heavy chain. If the
   canonical API instead stores a chain-level (not cell-level) object, the
   alignment contract must be restated. Documented per airr_context.md.
8. **MuData-vs-AnnData input.** The adapter duck-types both. Whether the public
   entry point accepts both or MuData-only (airr_context.md "Open checks") is
   still undecided.
9. **Licence gate on derived data.** Derived receptor CSVs are written to
   gitignored `data/external/` only; do not commit them until the processed
   Stephenson dataset's redistribution licence is confirmed
   (`manifest.license.processed_dataset: pending_verification`).
