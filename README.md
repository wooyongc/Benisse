![QBRC_logo](https://github.com/jcao89757/SCINA/blob/master/QBRC.jpg)
# BENISSE
## Introduction

Benisse (**B**CR **e**mbedding graphical **n**etwork **i**nformed by **s**cRNA-**S**eq) provides a powerful tool for analyzing B cell receptors (BCRs) guided by single cell gene expression of the B cells. As BCR affinity maturation is mainly controlled by signals obtained by B cells through BCRs with varying antigen binding strengths, B cells with similar BCR sequences are likely to have similar gene expression. Our deep contrastive learning model numerically embeds BCR CDR3H amino acid sequences, and Benisse employs a sparse graph learning model to capture the similarity between the BCRs and  single B cell expression.

Please refer to our paper for more details: TBA

Researchers searching for more bioinformatics tools please visit our lab website: https://qbrc.swmed.edu/labs/wanglab/index.php.

### Dependencies
Python (version 3.7), R (version 4.0.2), Linux (x86_64-redhat-linux-gn) shell (4.2.46(2) preferred)

**Python Packages**

pytorch (version 1.10.0), pandas (version 1.3.4), and sklearn (version 1.0), numpy (version 1.21.3)

**R Packages**

R UMAP (version 0.2.7.0) and Rtsne (version 0.15). Pseudotime inference was performed by [Monocle2](http://cole-trapnell-lab.github.io/monocle-release/docs/)

## Guided tutorial
In this tutorial, we will show a complete workflow for the Benisse. The toy example data we used in this tutorial are available on [github]() and on [figshare]().

### Installation
We recommend that you set up a python virtual environment for the BCR encoder using the commands below.
```{shell}
cd path/to/Benisse
# Following commands are used in high performance computing
# module purge
# module load shared
# module load python/3.7.x-anaconda
python3 -m venv ./environment
source environment/bin/activate
pip install torch
pip install pandas
pip install sklearn
pip install numpy
```

### Input data
1. BCR contig and heavy chain sequences in .csv format. Used by the BCR encoder. To be created by the user from the 10X contig file. The .csv should contain at least two columns in the names of "contigs" (unique identifiers of cells) and "cdr3" (BCR b-chain CDR3 sequences), or a folder path where contains all and only BCR sequence data files. Output will be concatenated into one output file.  Example: [10x_NSCLC.csv](https://github.com/wooyongc/Benisse/blob/main/example/10x_NSCLC.csv)

2. BCR contig file in .csv format. Easily adaptable from 10X software’s output. Used by the core Benisse model. Example: [10x_NSCLC_contigs.csv](https://github.com/wooyongc/Benisse/blob/main/example/10x_NSCLC_contig.csv)

3. Single B cell expression matrix in csv format. Used by the core Benisse model. See **suggested pre-processing workflow** for pre-processing. Also easily adaptable from 10X software's output. Example: [10x_NSCLC_exp.csv](https://github.com/wooyongc/Benisse/blob/main/example/10x_NSCLC_contig_exp.csv).

<img src="https://github.com/wooyongc/Benisse/blob/main/figs/10x_NSCLC.png" width="650">

**Fig.1 |** An example input of a BCR heavy chain sequence file


2. Single B cell expression matrix in .csv format. Easily adaptable from 10X software's output. Example: [10x_NSCLC_exp.csv](https://github.com/wooyongc/Benisse/blob/main/example/10x_NSCLC_exp.csv).

<img src="https://github.com/wooyongc/Benisse/blob/main/figs/10x_NSCLC_exp.png" width="650">

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
|model|The trained model to apply|
|output_data|The path to output .csv file|

Usage:
```{shell}
cd /path/to/Benisse
python3 AchillesEncoder.py \
--input_data example/10x_NSCLC.csv \
--output_data example/encoded_10x_NSCLC.csv
```
This script generates the numerical BCR embedding, which is used as an input in step 2. After the script finishes running, the embedded BCR sequence in .csv format will be generated using the **output** parameter. Example: [encoded_10x_NSCLC.csv](https://github.com/wooyongc/Benisse/blob/main/example/encoded_10x_NSCLC.csv)

<img src="https://github.com/wooyongc/Benisse/blob/main/figs/encoded_10x_NSCLC.png" width="650">

**Fig.3 |** An example BCR embedding generated

### Step 2: Run the core Benisse model
Using the generated BCR embedding, we can now run the R script, which is a sparse graph learner supervised on the single cell gene expression data. The following table describes the input parameters:

|Parameters|Description|
|----------|-------|
|expression|Single cell expression matrix. Please refer to 10x_NSCLC_exp.csv for format|
|contigs|The BCR sequence data. Please refer to 10x_NSCLC_contigs.csv for format|
|encoded BCR|The output file from AchillesEncoder.py|
|output|The output path directory|
|lambda2|Hyperparameter for Benisse|
|gamma|Hyperparameter for Benisse|
|max_iter|Maximum iteration for the model to run|
|lambda1|Hyperparameter for Benisse|
|rho|Hyperparameter for Benisse|
|m|Hyperparameter for Benisse|
|stop_cutoff|The variable is used to check the convergence. The algorithm stops if the mean squared distance between the sparse graph connection strength of the last 10 iterations is smaller than stop_cutoff|

The following example code runs the core Benisse model. We recommend using the hyperparameters in the following example to start with. For detailed explanation of the hyperparameters, please refer to Supplementary Note 1. of our paper.

```{r}
cd path/to/Benisse
Rscript Benisse.R \
example/10x_NSCLC_exp.csv \
example/10x_NSCLC_contigs.csv \
example/encoded_10x_NSCLC.csv \
example \
1610 1 100 1 1 10 1e-10
```

After the R script has been run, it will output the following files:

|Output|Description|
|----------|-------|
|clone_annotation.csv|Meta annotation information of the BCR clonotypes for each contigs, such as cluster size and graph labels|
|cleaned_exp.txt|Expression data filtered by attributes in the contigs file|
|clonality_label.txt|Corresponding relationship between B cells and B cell clonotypes|
|sparse_graph.txt|A Sparse graph built by Benisse, where each node represents the B cell clonotype and edge weights represent the similarity between BCRs.|
|latent_dist.txt|Latent distances between the BCR clonotypes, learned by Benisse through the supervision of gene expression information|
|connectionplot.pdf|Visualization of the graph representation of BCRs|
|in_cross_dist_check.pdf|Visualization of the BCR distances in the latent space|



<img src="https://github.com/wooyongc/Benisse/blob/main/figs/connectionplot.png" width="650">

**Fig.4 |** An example of connectionplot.pdf

<img src="https://github.com/wooyongc/Benisse/blob/main/figs/in_cross_dist_check_pdf.png" width="650">

**Fig.5 |** An example of in_cross_dist_check_pdf






## Version update
1.0.0: First release. (Nov 17th, 2021)

## Citation
<a id="1">[1]</a> Monocle2: Qiu, Xiaojie, et al. "Single-cell mRNA quantification and differential analysis with Census." Nature methods 14.3 (2017): 309-315.