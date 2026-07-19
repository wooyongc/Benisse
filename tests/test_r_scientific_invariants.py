import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
RSCRIPT = shutil.which("Rscript")


def run_r(expression):
    try:
        subprocess.run(
            [RSCRIPT, "-e", expression],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.CalledProcessError as error:
        pytest.fail(f"R component check failed:\n{error.stdout}\n{error.stderr}")


@pytest.mark.skipif(RSCRIPT is None, reason="Rscript is required for R component checks")
def test_convergence_norm_does_not_cancel_opposing_changes():
    run_r(
        """
source('R/util.R')
previous <- matrix(0, nrow=2, ncol=2)
current <- matrix(c(0, 1, -1, 0), nrow=2, byrow=TRUE)
legacy <- sum(current-previous)^2/nrow(current)^2
corrected <- graphChangeMSE(current, previous)
stopifnot(legacy == 0)
stopifnot(isTRUE(all.equal(corrected, 0.5)))
"""
    )


@pytest.mark.skipif(RSCRIPT is None, reason="Rscript is required for R component checks")
def test_committed_oracle_satisfies_scientific_invariants():
    run_r(
        """
load('example/Benisse_results.RData')
A <- results$A
Q <- results$Q
SI <- results$SI
latent <- as.matrix(read.table('example/latent_dist.txt'))
sparse <- as.matrix(read.table('example/sparse_graph.txt'))
tolerance <- 1e-10

stopifnot(all(is.finite(A)))
stopifnot(max(abs(A-t(A))) < tolerance)
stopifnot(max(abs(diag(A))) < tolerance)
stopifnot(min(A) >= -tolerance, max(A) <= 1+tolerance)
stopifnot(all(abs(A[SI == 0]) < tolerance))

stopifnot(all(is.finite(Q)))
stopifnot(max(abs(Q-t(Q))) < tolerance)
stopifnot(min(eigen(Q, symmetric=TRUE, only.values=TRUE)$values) > 0)

stopifnot(all(is.finite(latent)), min(latent) >= -tolerance)
stopifnot(max(abs(latent-t(latent))) < tolerance)
stopifnot(max(abs(diag(latent))) < tolerance)

stopifnot(max(abs(sparse-t(sparse))) < tolerance)
stopifnot(max(abs(diag(sparse))) < tolerance)
stopifnot(min(sparse) >= -tolerance, max(sparse) <= 1+tolerance)
stopifnot(all(sparse[SI == 0] == 0))
stopifnot(isTRUE(all.equal(sparse, A, check.attributes=FALSE)))
stopifnot(all((sparse > 0) == (results$sparse_graph == 1)))
"""
    )


@pytest.mark.skipif(RSCRIPT is None, reason="Rscript is required for R component checks")
def test_post_analysis_handles_graphs_with_fewer_than_fifty_directed_edges():
    run_r(
        """
source('R/post_analysis.R')
n <- 4
Q <- diag(n)
t <- matrix(c(0,0, 1,0, 0,1, 1,1), nrow=n, byrow=TRUE)
master <- as.matrix(dist(t))
sparse <- matrix(0, nrow=n, ncol=n)
sparse[1,2] <- sparse[2,1] <- 1
result <- testCor(
  master, Q, t, sparse,
  list(lambda1=1, lambda2=1, gamma=1, rho=1, m=2)
)
stopifnot(is.list(result), all(c('c1','c2') %in% names(result)))

empty <- testCor(
  master, Q, t, matrix(0, nrow=n, ncol=n),
  list(lambda1=1, lambda2=1, gamma=1, rho=1, m=2)
)
stopifnot(is.na(empty$c1), is.na(empty$c2))
"""
    )
