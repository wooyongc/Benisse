"""Internal Python post-analysis and plotting for Benisse results.

The functions consume the implementation-neutral ``BenisseNetworkResult``
contract. They do not select the R or corrected-Python numerical core and are
not yet a promised public plotting API.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

import airr_adapter as aa
import benisse_bridge as bridge


DISTANCE_LABELS = (
    "Connected in\nBCR networks",
    "Not connected,\nsharing V/J genes",
    "Not connected,\nnot sharing V/J genes",
)
PLOT_FILENAMES = {
    "connection": "python_connectionplot.pdf",
    "latent_distance": "python_latent_distance_groups.pdf",
    "network_summary": "python_network_summary.pdf",
    "coupling": "python_coupling_correlation.pdf",
}


@dataclass(frozen=True)
class BenissePlotData:
    network: aa.BenisseNetworkResult
    annotation: pd.DataFrame
    latent_distances: np.ndarray
    embedding: np.ndarray | None = None


def _validate_square_distances(
    values, n_nodes: int, name: str, *, require_zero_diagonal=True
) -> np.ndarray:
    matrix = np.atleast_2d(np.asarray(values, dtype="float64"))
    if matrix.shape != (n_nodes, n_nodes):
        raise ValueError(f"{name} shape {matrix.shape} != ({n_nodes}, {n_nodes})")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains non-finite values")
    if not np.allclose(matrix, matrix.T, rtol=0, atol=1e-8):
        raise ValueError(f"{name} must be symmetric")
    if require_zero_diagonal and not np.allclose(np.diag(matrix), 0, rtol=0, atol=1e-8):
        raise ValueError(f"{name} must have a zero diagonal")
    return matrix


def _read_embedding(encoded_csv, annotation: pd.DataFrame) -> np.ndarray:
    encoded = pd.read_csv(encoded_csv)
    if "index" not in encoded.columns:
        raise ValueError("encoded CSV is missing its CDR3 'index' column")
    feature_columns = [str(i) for i in range(20)]
    missing = set(feature_columns) - set(encoded.columns)
    if missing:
        raise ValueError(f"encoded CSV is missing feature columns {sorted(missing)}")
    if encoded["index"].duplicated().any():
        raise ValueError("encoded CSV contains ambiguous duplicate CDR3 values")
    lookup = encoded.set_index("index")[feature_columns]
    missing_cdr3 = sorted(set(annotation["cdr3"].astype(str)) - set(lookup.index.astype(str)))
    if missing_cdr3:
        raise ValueError(f"encoded CSV is missing {len(missing_cdr3)} annotated CDR3 values")
    matrix = lookup.loc[annotation["cdr3"].astype(str)].to_numpy(dtype="float64")
    if not np.isfinite(matrix).all():
        raise ValueError("encoded embedding contains non-finite values")
    return matrix


def read_plot_data(out_dir, *, encoded_csv=None) -> BenissePlotData:
    """Read standard Benisse outputs into a validated plotting bundle."""
    out_dir = Path(out_dir)
    network = bridge.read_benisse_network(out_dir)
    annotation = bridge.read_clone_annotation(out_dir)
    node_ids = bridge.clone_keys(annotation)
    if node_ids != network.node_ids:
        raise ValueError("annotation order does not match network node order")
    latent = _validate_square_distances(
        np.loadtxt(out_dir / bridge.OUTPUT_FILES["latent_dist"]),
        network.n_nodes,
        "latent distances",
    )
    embedding = _read_embedding(encoded_csv, annotation) if encoded_csv else None
    return BenissePlotData(network, annotation, latent, embedding)


def symmetric_adjacency(network: aa.BenisseNetworkResult, *, binary=False) -> sparse.csr_matrix:
    """Return the full symmetric adjacency represented by upper-triangle edges."""
    aa.validate_network_result(network)
    if network.directed:
        raise ValueError("Phase 2 plots currently require an undirected network")
    values = np.ones(network.n_edges) if binary else np.asarray(network.weight, dtype="float64")
    matrix = sparse.coo_matrix(
        (
            np.r_[values, values],
            (np.r_[network.row, network.col], np.r_[network.col, network.row]),
        ),
        shape=(network.n_nodes, network.n_nodes),
    )
    return matrix.tocsr()


def crude_graph(annotation: pd.DataFrame) -> np.ndarray:
    """Reconstruct R ``initiation.R``'s V/J-family candidate graph."""
    required = {"v_gene", "j_gene"}
    missing = required - set(annotation.columns)
    if missing:
        raise ValueError(f"annotation missing columns {sorted(missing)}")
    families = (
        annotation["v_gene"].fillna("").astype(str)
        + "\x1f"
        + annotation["j_gene"].fillna("").astype(str)
    ).to_numpy()
    candidate = families[:, None] == families[None, :]
    np.fill_diagonal(candidate, False)
    return candidate


def component_labels(network: aa.BenisseNetworkResult) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic connected-component labels and their sizes."""
    adjacency = symmetric_adjacency(network, binary=True)
    count, raw = connected_components(adjacency, directed=False)
    sizes = np.bincount(raw, minlength=count)
    order = sorted(range(count), key=lambda i: (-sizes[i], int(np.flatnonzero(raw == i)[0])))
    remap = np.empty(count, dtype="int64")
    remap[order] = np.arange(count)
    labels = remap[raw]
    return labels, np.bincount(labels, minlength=count)


def pca_coordinates(embedding) -> np.ndarray:
    """Project a node embedding to two deterministic principal-component axes."""
    matrix = np.asarray(embedding, dtype="float64")
    if matrix.ndim != 2 or matrix.shape[0] < 1:
        raise ValueError("embedding must be a non-empty two-dimensional matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("embedding contains non-finite values")
    if matrix.shape[0] == 1:
        return np.zeros((1, 2), dtype="float64")
    n_components = min(2, matrix.shape[0], matrix.shape[1])
    coordinates = PCA(n_components=n_components, svd_solver="full").fit_transform(matrix)
    if n_components == 1:
        coordinates = np.column_stack([coordinates[:, 0], np.zeros(matrix.shape[0])])
    for axis in range(2):
        pivot = int(np.argmax(np.abs(coordinates[:, axis])))
        if coordinates[pivot, axis] < 0:
            coordinates[:, axis] *= -1
    return coordinates


def embedding_coordinates(embedding, *, method="pca", random_state=0) -> np.ndarray:
    """Return deterministic PCA, UMAP, or t-SNE node coordinates."""
    matrix = np.asarray(embedding, dtype="float64")
    if method == "pca":
        return pca_coordinates(matrix)
    if matrix.ndim != 2 or matrix.shape[0] < 3 or not np.isfinite(matrix).all():
        raise ValueError(f"{method} requires at least three finite embedding rows")
    if method == "umap":
        if "NUMBA_CACHE_DIR" not in os.environ:
            cache_dir = Path(tempfile.gettempdir()) / "benisse-numba-cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            os.environ["NUMBA_CACHE_DIR"] = str(cache_dir)
        try:
            from umap import UMAP
        except ImportError as exc:
            raise ImportError("UMAP plotting requires umap-learn") from exc
        return UMAP(n_components=2, random_state=random_state).fit_transform(matrix)
    if method == "tsne":
        from sklearn.manifold import TSNE

        perplexity = min(30.0, max(2.0, (matrix.shape[0] - 1) / 3))
        return TSNE(
            n_components=2,
            init="pca",
            learning_rate="auto",
            perplexity=perplexity,
            random_state=random_state,
        ).fit_transform(matrix)
    raise ValueError("method must be pca, umap, or tsne")


def latent_distance_groups(data: BenissePlotData) -> dict[str, np.ndarray]:
    """Partition latent distances by learned-edge and candidate-graph status."""
    n = data.network.n_nodes
    latent = _validate_square_distances(data.latent_distances, n, "latent distances")
    learned = symmetric_adjacency(data.network, binary=True).toarray().astype(bool)
    candidate = crude_graph(data.annotation)
    upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    masks = (
        learned & upper,
        ~learned & candidate & upper,
        ~candidate & upper,
    )
    return {label: latent[mask] for label, mask in zip(DISTANCE_LABELS, masks)}


def edge_retention_ratio(data: BenissePlotData) -> float:
    candidates = int(np.count_nonzero(np.triu(crude_graph(data.annotation), k=1)))
    if candidates == 0:
        return float("nan")
    return data.network.n_edges / candidates


def clone_expression_distances(cleaned_exp, clonality_labels, node_ids) -> np.ndarray:
    """Reproduce ``initiation.R``'s clone-level expression-distance matrix.

    ``cleaned_exp`` is genes x cells and ``clonality_labels`` has one clone key
    per cell. The calculation selects the most variable expression decile,
    performs the same unscaled 10-PC projection, sums cell distances by clone,
    and applies the legacy normalization.
    """
    if isinstance(cleaned_exp, (str, Path)):
        expression = pd.read_csv(cleaned_exp, sep=r"\s+", index_col=0)
    else:
        expression = pd.DataFrame(cleaned_exp).copy()
    if isinstance(clonality_labels, (str, Path)):
        labels = pd.read_csv(clonality_labels, header=None).iloc[:, 0].astype(str).to_numpy()
    else:
        labels = np.asarray(clonality_labels, dtype=str)
    values = expression.to_numpy(dtype="float64")
    if values.ndim != 2 or values.shape[1] != labels.size:
        raise ValueError("expression columns and clonality labels must have equal length")
    if labels.size < 2 or not np.isfinite(values).all():
        raise ValueError("expression data require at least two finite cells")
    deviations = np.std(values, axis=1, ddof=1)
    keep = deviations > np.quantile(deviations, 0.9)
    if np.count_nonzero(keep) < 1:
        raise ValueError("no genes remain after variable-gene selection")
    cell_matrix = values[keep].T
    n_components = min(10, cell_matrix.shape[0], cell_matrix.shape[1])
    scores = PCA(n_components=n_components, svd_solver="full").fit_transform(cell_matrix)
    cell_distances = squareform(pdist(scores))

    unique_labels = np.asarray(sorted(set(labels)))
    indicator = sparse.csr_matrix(
        (
            np.ones(labels.size),
            (np.arange(labels.size), np.searchsorted(unique_labels, labels)),
        ),
        shape=(labels.size, unique_labels.size),
    )
    aggregated = np.asarray(indicator.T @ cell_distances @ indicator)
    scaler = 4 * aggregated.sum() / (labels.size * (labels.size - 1))
    if scaler <= 0 or not np.isfinite(scaler):
        raise ValueError("expression-distance scaling is zero or non-finite")
    clone_sizes = np.asarray(indicator.sum(axis=0)).ravel()
    normalized = aggregated / clone_sizes[:, None] / clone_sizes[None, :] / scaler
    positions = {name: i for i, name in enumerate(unique_labels)}
    missing = [name for name in node_ids if name not in positions]
    if missing:
        raise ValueError(f"clonality labels are missing {len(missing)} graph nodes")
    order = np.asarray([positions[name] for name in node_ids])
    result = normalized[np.ix_(order, order)]
    return (result + result.T) / 2


def bcr_embedding_distances(embedding) -> np.ndarray:
    matrix = np.asarray(embedding, dtype="float64")
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("embedding must be a finite two-dimensional matrix")
    return squareform(pdist(matrix))


def coupling_data(
    expression_distances,
    latent_distances,
    embedding_distances,
    network: aa.BenisseNetworkResult,
    *,
    target_points=300,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Reproduce the data reduction and Spearman statistics in R ``testCor``."""
    if target_points < 1:
        raise ValueError("target_points must be positive")
    n = network.n_nodes
    matrices = [
        _validate_square_distances(
            matrix, n, name, require_zero_diagonal=(name != "expression distances")
        )
        for matrix, name in (
            (expression_distances, "expression distances"),
            (latent_distances, "latent distances"),
            (embedding_distances, "embedding distances"),
        )
    ]
    learned = symmetric_adjacency(network, binary=True).toarray().astype(bool)
    if not learned.any():
        empty = pd.DataFrame(columns=["expression", "latent", "embedding"])
        return empty, {"expression_latent": float("nan"), "expression_embedding": float("nan")}
    values = [matrix[learned] for matrix in matrices]
    order = np.argsort(values[0], kind="stable")
    values = [value[order] for value in values]
    group_size = len(order) // target_points
    if group_size < 5:
        group_size = len(order) // 50
    group_size = max(1, group_size)
    full_groups = len(order) // group_size
    groups = np.repeat(np.arange(full_groups), group_size)
    if groups.size < len(order):
        # R appends the remainder to its final complete group rather than
        # creating a smaller trailing group.
        groups = np.r_[groups, np.repeat(full_groups - 1, len(order) - groups.size)]
    frame = pd.DataFrame(
        {"expression": values[0], "latent": values[1], "embedding": values[2], "group": groups}
    )
    reduced = frame.groupby("group", sort=True).median(numeric_only=True).reset_index(drop=True)
    correlations = {
        "expression_latent": float(spearmanr(reduced["expression"], reduced["latent"]).statistic),
        "expression_embedding": float(
            spearmanr(reduced["expression"], reduced["embedding"]).statistic
        ),
    }
    return reduced, correlations


def _node_colors(data: BenissePlotData, color_by: str):
    annotation = data.annotation
    if color_by == "clone_size":
        if "clsize" not in annotation:
            raise ValueError("annotation has no clsize column")
        values = pd.to_numeric(annotation["clsize"], errors="raise").to_numpy(dtype="float64")
        return np.log2(values + 1), "viridis", "log2(clone size + 1)"
    if color_by == "vj_family":
        labels = annotation["v_gene"].astype(str) + " / " + annotation["j_gene"].astype(str)
    elif color_by == "graph_component":
        components, sizes = component_labels(data.network)
        labels = pd.Series(
            ["singleton" if sizes[item] == 1 else f"component {item + 1}" for item in components]
        )
    else:
        raise ValueError("color_by must be clone_size, vj_family, or graph_component")
    codes, _ = pd.factorize(labels, sort=True)
    return codes, "tab20", color_by.replace("_", " ")


def plot_embedding_network(
    data: BenissePlotData, *, color_by="clone_size", method="pca", random_state=0, ax=None
):
    """Plot encoder-PC coordinates and learned graph edges."""
    if data.embedding is None:
        raise ValueError("an encoder embedding is required for the connection plot")
    coordinates = embedding_coordinates(
        data.embedding, method=method, random_state=random_state
    )
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
    segments = np.stack([coordinates[data.network.row], coordinates[data.network.col]], axis=1)
    if len(segments):
        widths = 0.25 + 0.75 * np.asarray(data.network.weight) / max(data.network.weight)
        ax.add_collection(LineCollection(segments, colors="0.72", linewidths=widths, alpha=0.45))
    colors, cmap, color_label = _node_colors(data, color_by)
    clone_sizes = (
        pd.to_numeric(data.annotation["clsize"], errors="coerce").fillna(1).to_numpy()
        if "clsize" in data.annotation else np.ones(data.network.n_nodes)
    )
    sizes = 9 + 12 * np.sqrt(clone_sizes.astype("float64"))
    points = ax.scatter(
        coordinates[:, 0], coordinates[:, 1], c=colors, cmap=cmap, s=sizes,
        alpha=0.8, edgecolors="none", zorder=2,
    )
    if color_by == "clone_size":
        ax.figure.colorbar(points, ax=ax, shrink=0.72, label=color_label)
    axis_prefix = {"pca": "PC", "umap": "UMAP", "tsne": "t-SNE"}[method]
    ax.set(
        xlabel=f"Encoder {axis_prefix}1",
        ylabel=f"Encoder {axis_prefix}2",
        title="Benisse clonotype network",
    )
    ax.spines[["top", "right"]].set_visible(False)
    return ax


def plot_latent_distance_groups(data: BenissePlotData, *, ax=None):
    """Plot the Python equivalent of R ``checkDist`` using unique node pairs."""
    groups = latent_distance_groups(data)
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
    values = [groups[label] for label in DISTANCE_LABELS]
    positions = np.arange(1, 4)
    box = ax.boxplot(values, positions=positions, widths=0.5, patch_artist=True, showfliers=False)
    palette = plt.get_cmap("Dark2")(np.arange(3))
    for patch, color in zip(box["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    ax.set_xticks(positions, DISTANCE_LABELS, rotation=20, ha="right")
    ax.set_ylabel("BCR distances in latent space")
    ax.set_title("Latent distance by graph relationship")
    ax.spines[["top", "right"]].set_visible(False)
    return ax


def plot_network_summary(data: BenissePlotData, *, axes=None):
    """Plot clone-size and connected-component diagnostics."""
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(8, 3.8), constrained_layout=True)
    axes = np.asarray(axes).ravel()
    if axes.size != 2:
        raise ValueError("network summary requires exactly two axes")
    clone_sizes = (
        pd.to_numeric(data.annotation["clsize"], errors="coerce").fillna(1)
        if "clsize" in data.annotation
        else pd.Series(np.ones(data.network.n_nodes))
    )
    clone_counts = clone_sizes.value_counts().sort_index()
    axes[0].bar(clone_counts.index, clone_counts.values, color=plt.get_cmap("Dark2")(0))
    axes[0].set(xlabel="Clone size (cells)", ylabel="Clonotypes", title="Clone-size distribution")
    _, sizes = component_labels(data.network)
    component_counts = pd.Series(sizes).value_counts().sort_index()
    axes[1].bar(component_counts.index, component_counts.values, color=plt.get_cmap("Dark2")(1))
    ratio = edge_retention_ratio(data)
    ratio_text = "n/a" if not np.isfinite(ratio) else f"{ratio:.1%}"
    axes[1].set(
        xlabel="Nodes per component", ylabel="Components",
        title=f"Network components\n{data.network.n_edges} edges; {ratio_text} of V/J candidates",
    )
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    return axes


def plot_coupling_correlation(reduced: pd.DataFrame, correlations: dict[str, float], *, ax=None):
    """Plot expression distance against latent and original-embedding distance."""
    required = {"expression", "latent", "embedding"}
    if required - set(reduced.columns):
        raise ValueError(f"reduced coupling data must contain {sorted(required)}")
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
    ax.scatter(reduced["expression"], reduced["latent"], s=18, alpha=0.7, label="Latent")
    ax.scatter(
        reduced["expression"], reduced["embedding"], s=18, alpha=0.7,
        marker="^", label="Encoder",
    )
    ax.set(
        xlabel="Clone expression distance", ylabel="BCR distance",
        title=(
            "Expression–BCR coupling\n"
            f"Spearman latent={correlations['expression_latent']:.3f}; "
            f"encoder={correlations['expression_embedding']:.3f}"
        ),
    )
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    return ax


def generate_post_analysis_plots(out_dir, encoded_csv, destination=None) -> dict[str, Path]:
    """Generate Python PDFs from a standard completed Benisse output directory."""
    out_dir = Path(out_dir)
    destination = Path(destination) if destination else out_dir
    destination.mkdir(parents=True, exist_ok=True)
    data = read_plot_data(out_dir, encoded_csv=encoded_csv)
    generated = {}
    for key, plotter in (
        ("connection", plot_embedding_network),
        ("latent_distance", plot_latent_distance_groups),
        ("network_summary", plot_network_summary),
    ):
        plotter(data)
        path = destination / PLOT_FILENAMES[key]
        plt.gcf().savefig(path, bbox_inches="tight")
        plt.close(plt.gcf())
        generated[key] = path

    cleaned = out_dir / bridge.OUTPUT_FILES["cleaned_exp"]
    clonality = out_dir / bridge.OUTPUT_FILES["clonality_label"]
    if cleaned.exists() and clonality.exists():
        expression = clone_expression_distances(cleaned, clonality, data.network.node_ids)
        reduced, correlations = coupling_data(
            expression,
            data.latent_distances,
            bcr_embedding_distances(data.embedding),
            data.network,
        )
        plot_coupling_correlation(reduced, correlations)
        path = destination / PLOT_FILENAMES["coupling"]
        plt.gcf().savefig(path, bbox_inches="tight")
        plt.close(plt.gcf())
        generated["coupling"] = path
    return generated
