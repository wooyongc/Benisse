[![DOI](https://zenodo.org/badge/429241506.svg)](https://zenodo.org/badge/latestdoi/429241506)

![QBRC_logo](https://github.com/jcao89757/SCINA/blob/master/QBRC.jpg)


# BENISSE
## Introduction

Benisse (**B**CR **e**mbedding graphical **n**etwork **i**nformed by **s**cRNA-**S**eq) provides a powerful tool for analyzing B cell receptors (BCRs) guided by single cell gene expression of the B cells. As BCR affinity maturation is controlled by signals obtained by B cells through the BCRs with varying antigen binding strengths, B cells with similar BCR sequences are likely to have similar transcriptomic profiles. Our deep contrastive learning model numerically embeds BCR CDR3H amino acid sequences, and Benisse employs a sparse graph learning model to capture the mutual information shared between the BCRs and single B cell expression, based upon the BCR encoder.

Please refer to our paper for more details : [Interpreting the B-cell receptor repertoire with single-cell gene expression using Benisse](https://www.nature.com/articles/s42256-022-00492-6)

Researchers searching for more immunology-related bioinformatics tools can visit our lab website: https://qbrc.swmed.edu/labs/wanglab/index.php.

### Dependencies
Python (version 3.10). R 4.3 is optional and retained only for frozen v1 scientific-oracle tests;
the corrected v2 runtime is Python-only.

**Python Packages**

pytorch (version 2.2.2, CPU build), pandas (version 2.3.3), scikit-learn, and numpy (version 1.26.4)

**Optional legacy-oracle R Packages**

ggplot2 (version 3.5.1; requires >= 3.4.0), data.table (version 1.16.0), and igraph (version 2.1.4; requires >= 2.0.0).

## Guided tutorial
In this tutorial, we will show a complete workflow for Benisse. The toy example data we used in this tutorial are available on [github](https://github.com/wooyongc/Benisse/tree/main/example) and on [figshare](https://doi.org/10.6084/m9.figshare.17035931).

### Installation
We recommend a conda environment for the BCR encoder using the commands below.
```{shell}
# Navigate to your directory of preference
cd path/to/workdir

# Clone the repository
git clone https://github.com/wooyongc/Benisse.git

# Navigate to the Benisse directory
cd Benisse

# Create and activate a conda environment (Python 3.10)
conda create -n benisse python=3.10 -y
conda activate benisse

# Install dependencies.
# Note: install `scikit-learn`, not the deprecated `sklearn` stub on PyPI.
# torch 2.2.2 is the last version with Intel-macOS (x86_64) CPU wheels.
pip install "torch==2.2.2" "pandas==2.3.3" "numpy==1.26.4" scikit-learn

# Deactivate the environment when you are done with the analyses
conda deactivate
```

For the v2 AnnData/AIRR integration work, an isolated Scirpy compatibility environment is
also recorded in `environment-scirpy022.yml`:

```{shell}
conda env create -f environment-scirpy022.yml
conda activate benisse-scirpy022
```

Downloaded AIRR reference objects remain outside git. Their source, integrity hashes,
structure, local fixture derivation, and redistribution status are recorded in
`data/manifest.yaml`.

Installation time will be about 30min, depending on the computing system

### Input data
1. BCR contig and heavy chain sequences in .csv format. Used by the BCR encoder. To be created by the user from the 10X contig file (input file 2 below). The .csv should contain at least two columns in the names of "contigs" (unique identifiers of cells) and "cdr3" (BCR CDR3H sequences), or a folder path which contains all and only the BCR sequence data files. Output will be concatenated into one output file.  Example: [10x_NSCLC.csv](https://github.com/wooyongc/Benisse/blob/main/example/10x_NSCLC.csv)

2. BCR contig file in .csv format. Easily adaptable from 10X software’s output. Used by the core Benisse model. Example: [10x_NSCLC_contigs.csv](https://github.com/wooyongc/Benisse/blob/main/example/10x_NSCLC_contigs.csv)

3. Single B cell expression matrix in csv format. Used by the core Benisse model. See **suggested pre-processing workflow** for pre-processing. Also easily adaptable from 10X software's output. Example: [10x_NSCLC_exp.csv](https://github.com/wooyongc/Benisse/blob/main/example/10x_NSCLC_exp.csv).

<img src="https://github.com/wooyongc/Benisse/blob/main/figs/10x_NSCLC.png" width="700">

**Fig.1 |** An example input of a BCR heavy chain sequence file

<img src="https://github.com/wooyongc/Benisse/blob/main/figs/10x_NSCLC_exp.png" width="700">

**Fig.2 |** An example of B cell expression matrix


### Suggested pre-processing workflow
Double check your 'contig_id' column, to make sure all IDs are unique. The 'cdr3' column should not have any letters, numbers or symbols that do not represent amino acids.

For the single cell expression data, log-transformation (log(x+1)) is always suggested. Users are encouraged to use their own preferred method to normalize the data. Some useful papers are listed below for you to refer to.
1. [Hafemeister, Christoph, and Rahul Satija. "Normalization and variance stabilization of single-cell RNA-seq data using regularized negative binomial regression." Genome Biology 20, no. 1 (2019): 1-15.](https://link.springer.com/article/10.1186/s13059-019-1874-1)
2. [Vallejos, Catalina A., Davide Risso, Antonio Scialdone, Sandrine Dudoit, and John C. Marioni. "Normalizing single-cell RNA sequencing data: challenges and opportunities." Nature methods 14, no. 6 (2017): 565.](https://www.nature.com/articles/nmeth.4292)

Moreover, the expression matrix can be replaced with the results from any dimension reduction (DR) method, for example, PCA, t-SNE, or UMAP, as long as the DR format matches the original expression matrix (coordinates or features on the rows, and cell identifiers on the columns).

### Step 1: Generating BCR embedding

The numerical BCR embedding script takes the following as input parameters:

|Parameters|Description|
|----------|-------|
|input_data|A .csv file OR folder containing the BCR sequence data to be embedded|
|output_data|The path to the output .csv file|
|cuda| Whether to utilize CUDA or not. `--cuda False` if you are using CPU|

Usage:
```{shell}
# Navigate to the path you installed Benisse
cd /path/to/Benisse

# Activate the conda environment
conda activate benisse

# Run the Encoder (use --cuda False on machines without a CUDA GPU)
python3 AchillesEncoder.py \
--input_data example/10x_NSCLC.csv \
--output_data example/encoded_10x_NSCLC.csv \
--cuda False

# Deactivate the environment when you are done with the analyses
conda deactivate
```
This script generates the numerical BCR embeddings, which is used as an input in step 2. After the script finishes running, the embedded BCR sequence in .csv format will be generated using the **output** parameter. Example: [encoded_10x_NSCLC.csv](https://github.com/wooyongc/Benisse/blob/main/example/encoded_10x_NSCLC.csv)

<img src="https://github.com/wooyongc/Benisse/blob/main/figs/encoded_10x_NSCLC.png" width="700">

**Fig.3 |** An example BCR embedding generated

### Step 2: Run the corrected v2 Benisse model
Using the generated BCR embedding, the corrected Python sparse-graph learner is supervised on
the single-cell expression data. The following table describes the parameters:

|Parameters|Description|
|----------|-------|
|expression|Single cell expression matrix. Please refer to 10x_NSCLC_exp.csv for format|
|contigs|The BCR sequence data. Please refer to 10x_NSCLC_contigs.csv for format|
|encoded BCR|The output file from AchillesEncoder.py|
|output|The output path directory|
|lambda2|Hyperparameter for Benisse. Penalty strength for considering the single cell expression data in the generation of the BCR latent embedding. |
|gamma|Hyperparameter for Benisse. Penalty term for the prior distribution of the latent space embedding. |
|max_iter|Maximum iteration for the model to run|
|lambda1|Hyperparameter for Benisse. Slack variable penalty for controlling for noise in the embedding. |
|rho|Hyperparameter for Benisse. A parameter for executing the alternating direction method of multipliers algorithm. See Sup. Note 1 for details. |
|m|Hyperparameter for Benisse. Dimension of the latent space. |
|stop_cutoff|The variable is used to check the convergence. The algorithm stops if the mean squared distance between the sparse graph connection strength of the last 10 iterations is smaller than stop_cutoff|

The following development-branch example runs the complete encoder, corrected core, and Python
plots. We recommend these paper-example hyperparameters as a starting point. Packaging and the
final CLI are intentionally deferred until this scientific workflow stabilizes.

```python
from benisse_pipeline import BenisseParams, run_csv_pipeline

result = run_csv_pipeline(
    "example/10x_NSCLC.csv",
    "example/10x_NSCLC_exp.csv",
    "example/10x_NSCLC_contigs.csv",
    "/tmp/benisse-v2-example",
    params=BenisseParams(),
    cuda=False,
)
print(result.network.n_nodes, result.network.n_edges)
```

The corrected v2 example has 1,494 nodes and 1,592 edges. The frozen v1 R result has 1,691
edges because its undirected `A` update uses an inconsistent directed parameterization and
Jacobian. This is an intentional scientific migration documented in `PHASE4_NATIVE_NOTES.md`.

For MuData/AIRR input, `run_mudata_pipeline(...)` performs deterministic productive-heavy-chain
selection, attaches cell embeddings to `mdata.mod["airr"].obsm["X_benisse"]`, and stores the
clone network under `mdata.uns["benisse"]`.

The corrected Python pipeline writes compatible scientific tables plus Python-native plots:

|Output|Description|
|----------|-------|
|clone_annotation.csv|Meta annotation information of the BCR clonotypes, such as graph node labels and clone sizes|
|cleaned_exp.txt|Expression data of the B cells|
|clonality_label.txt|Corresponding relationship between individual B cells and B cell clonotypes|
|sparse_graph.txt|A sparse graph built by Benisse, where each node represents a B cell clonotype and edge weights represent the similarity between BCRs (higher is more similar)|
|latent_dist.txt|Distances between the BCR clonotypes in the latent space, learned by Benisse through the supervision of gene expression information|
|python_connectionplot.pdf|Visualization of encoder coordinates and corrected graph edges|
|python_latent_distance_groups.pdf|Latent BCR distances grouped by graph relationship|
|python_network_summary.pdf|Clone-size, component-size, and retained-candidate-edge diagnostics|
|python_coupling_correlation.pdf|Expression–BCR coupling scatter and Spearman statistics|

### Frozen v1 R oracle

`Benisse.R`, its historical outputs, and the Phase 4c bridge remain available to reproduce and
test the published behavior. They are no longer the v2 runtime backend. Do not rebaseline v1
outputs to corrected-v2 topology.



<img src="https://github.com/wooyongc/Benisse/blob/main/figs/connectionplot.png" width = "500">

**Fig.4 |** An example of connectionplot.pdf

<img src="https://github.com/wooyongc/Benisse/blob/main/figs/in_cross_dist_check.png">

**Fig.5 |** An example of in_cross_dist_check_pdf



The runtime on our example dataset should be <30min for both steps combined

## Scientific parity tests

The Phase 4 modernization is guarded by a pytest suite. Run the fast CPU checks with:

```{shell}
conda run -n benisse-scirpy022 python -m pytest -v
```

Run the explicit full Python + R oracle check with:

```{shell}
BENISSE_RUN_SLOW_TESTS=1 conda run -n benisse-scirpy022 \
  python -m pytest -m slow -v -s
```

The slow check compares the encoded CSV and stable R outputs byte-for-byte, compares the
RData result semantically with exact sparse-edge agreement, and compares rasterized PDF pages.
See `tests/README.md` for details.


## Version update
1.0.0: First release. (Nov 17th, 2021)

## Citation
<a id="1">[1]</a> Monocle2: Qiu, Xiaojie, et al. "Single-cell mRNA quantification and differential analysis with Census." Nature methods 14.3 (2017): 309-315.
