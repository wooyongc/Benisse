# Phase 2 Python post-analysis and plots

Status: implemented as an **internal, implementation-neutral plotting layer** on
`feat/python-post-analysis-plots`. It consumes the shared `BenisseNetworkResult` contract and
therefore works with the supported Phase 4c R bridge or the experimental corrected Python core.
It does not change which numerical implementation is the default.

## What was delivered

- `benisse_plotting.py`:
  - validates and aligns the network, clone annotation, latent-distance matrix, and optional
    20-dimensional encoder embedding;
  - reconstructs the V/J-family candidate graph used by `R/initiation.R`;
  - calculates true connected components and the retained-candidate-edge ratio;
  - reproduces the clone-level expression-distance calculation from `R/initiation.R`, including
    variable-gene selection, ten-PC cell distances, clone aggregation, and legacy scaling;
  - reproduces `R/post_analysis.R::testCor` median aggregation and Spearman statistics, including
    its directed-edge convention and remainder-bin behavior;
  - plots the encoder embedding with graph edges using PCA, optional deterministic UMAP, or
    deterministic t-SNE coordinates;
  - colors nodes by clone size, V/J family, or learned graph component;
  - plots latent distances by graph relationship, clone-size/component diagnostics, edge
    retention, and expression–BCR coupling;
  - writes four Python PDFs with `python_` prefixes so the committed R oracle plots are never
    overwritten.
- `tests/test_benisse_plotting.py` covers input validation, graph reconstruction, components,
  pair classification, deterministic projection, clone-expression aggregation, correlations,
  empty graphs, all node-color modes, PDF rendering, and the complete committed example.
- `tests/fixtures/export_post_analysis_oracle.R` exports only the R quantities required for
  scientific parity checks; it does not alter production R code or committed outputs.

## Parity decisions

- Learned undirected edges and latent-distance categories are plotted once per unique node pair.
  R duplicates symmetric pairs; removing the duplicate affects rendering cost, not the
  distributions.
- Coupling statistics intentionally retain R's directed duplication and exact bin construction,
  because changing those details changes the reported Spearman values.
- Clone-level expression matrices can have nonzero diagonals: expanded clones aggregate
  distances between distinct cells belonging to the same clone. This is validated rather than
  incorrectly forced to ordinary zero-diagonal distance semantics.
- Graph-component colors use SciPy's actual connected components. The old `graph_label` column
  remains readable as annotation, but is not trusted as the component algorithm.
- The default layout remains PCA for direct continuity with the R plot. UMAP and t-SNE are
  opt-in views and must not be interpreted as numerical-core outputs.

## Generate plots

After a standard Benisse run:

```python
import benisse_plotting as bp

paths = bp.generate_post_analysis_plots(
    out_dir="example",
    encoded_csv="example/encoded_10x_NSCLC.csv",
    destination="/tmp/benisse-python-plots",
)
```

This produces:

- `python_connectionplot.pdf`
- `python_latent_distance_groups.pdf`
- `python_network_summary.pdf`
- `python_coupling_correlation.pdf` when cleaned expression and clonality labels are present

The module is still internal. Public names, Scanpy-style plotting conventions, return types,
and CLI exposure remain Phase 4e decisions.
