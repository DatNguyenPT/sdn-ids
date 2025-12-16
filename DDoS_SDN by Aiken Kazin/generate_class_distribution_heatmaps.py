#!/usr/bin/env python3
"""
Generate class-distribution heatmaps for Federated Learning partitions (IID / Non-IID).

This script is intentionally **standalone** (no training). It reads labels from a CSV,
simulates disjoint FL client partitions across N workers, optionally groups workers into
"controllers", then saves heatmap PNG images.

Examples:
  # Generate both IID + Non-IID heatmaps (workers + controllers) into ./visualizations/
  python generate_class_distribution_heatmaps.py --dataset-csv dataset_sdn.csv --n-workers 8 --n-controllers 2

  # Only non-IID (binary alternating skew), no controllers
  python generate_class_distribution_heatmaps.py --partition noniid --n-workers 6 --n-controllers 0

Notes:
  - IID partition uses stratified disjoint splitting (each worker receives ~same class ratios).
  - Non-IID partition (default) creates class-skewed clients:
      * Binary case (0/1): odd workers majority class 0, even workers majority class 1
      * Multi-class: worker i majority class = i % n_classes
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class LabelEncoding:
    """Mapping between raw labels and contiguous integer ids [0..K-1]."""

    raw_to_id: Dict[str, int]
    id_to_raw: List[str]


def _read_labels_from_csv(dataset_csv: str, label_col: str) -> Tuple[np.ndarray, LabelEncoding]:
    """Read label column from CSV without pandas (fast + portable)."""
    if not os.path.exists(dataset_csv):
        raise FileNotFoundError(f"CSV not found: {dataset_csv}")

    raw_labels: List[str] = []
    with open(dataset_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or label_col not in reader.fieldnames:
            raise ValueError(
                f"Label column '{label_col}' not found in CSV header. "
                f"Available: {reader.fieldnames}"
            )
        for row in reader:
            raw_labels.append(str(row[label_col]).strip())

    # If labels are already integers (e.g., "0"/"1"), keep them stable and sorted
    all_int = True
    ints: List[int] = []
    for x in raw_labels:
        try:
            ints.append(int(float(x)))  # allow "0.0"
        except Exception:
            all_int = False
            break

    if all_int:
        uniq = sorted(set(ints))
        raw_to_id = {str(v): i for i, v in enumerate(uniq)}
        id_to_raw = [str(v) for v in uniq]
        y = np.array([raw_to_id[str(v)] for v in ints], dtype=np.int64)
        return y, LabelEncoding(raw_to_id=raw_to_id, id_to_raw=id_to_raw)

    # Otherwise, build a stable encoding from sorted unique raw strings
    uniq_raw = sorted(set(raw_labels))
    raw_to_id = {v: i for i, v in enumerate(uniq_raw)}
    id_to_raw = uniq_raw
    y = np.array([raw_to_id[v] for v in raw_labels], dtype=np.int64)
    return y, LabelEncoding(raw_to_id=raw_to_id, id_to_raw=id_to_raw)


def _train_split_indices(y: np.ndarray, train_frac: float, seed: int) -> np.ndarray:
    """Stratified train split indices (no sklearn)."""
    if not (0.0 < train_frac < 1.0):
        raise ValueError("train_frac must be in (0, 1)")
    rng = np.random.RandomState(seed)

    train_indices: List[int] = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_train_c = int(round(len(idx) * train_frac))
        train_indices.extend(idx[:n_train_c].tolist())

    train_indices = np.array(train_indices, dtype=np.int64)
    rng.shuffle(train_indices)
    return train_indices


def _allocate_worker_sizes(n_samples: int, n_workers: int) -> List[int]:
    """Evenly distribute n_samples across n_workers (sum preserved)."""
    base = n_samples // n_workers
    rem = n_samples % n_workers
    return [base + (1 if i < rem else 0) for i in range(n_workers)]


def _partition_iid(y: np.ndarray, n_workers: int, seed: int) -> List[np.ndarray]:
    """Disjoint, stratified IID partitions."""
    rng = np.random.RandomState(seed)
    classes = np.unique(y)

    # For each class, split its indices into n_workers chunks
    class_chunks: Dict[int, List[np.ndarray]] = {}
    for c in classes:
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        sizes = _allocate_worker_sizes(len(idx), n_workers)
        chunks = []
        start = 0
        for s in sizes:
            chunks.append(idx[start : start + s])
            start += s
        class_chunks[int(c)] = chunks

    worker_indices: List[List[int]] = [[] for _ in range(n_workers)]
    for w in range(n_workers):
        for c in classes:
            worker_indices[w].extend(class_chunks[int(c)][w].tolist())
        rng.shuffle(worker_indices[w])

    return [np.array(ix, dtype=np.int64) for ix in worker_indices]


def _partition_noniid(y: np.ndarray, n_workers: int, seed: int, skew: float) -> List[np.ndarray]:
    """Disjoint, class-skewed Non-IID partitions."""
    if not (0.5 <= skew <= 1.0):
        raise ValueError("skew must be in [0.5, 1.0]")
    rng = np.random.RandomState(seed)
    classes = np.unique(y)
    n_classes = len(classes)

    # Pools of remaining indices per class
    pools: Dict[int, List[int]] = {}
    for c in classes:
        idx = np.where(y == c)[0].tolist()
        rng.shuffle(idx)
        pools[int(c)] = idx

    worker_sizes = _allocate_worker_sizes(len(y), n_workers)
    worker_indices: List[List[int]] = [[] for _ in range(n_workers)]

    def pop_many(pool: List[int], k: int) -> List[int]:
        if k <= 0:
            return []
        k = min(k, len(pool))
        out = pool[:k]
        del pool[:k]
        return out

    for w in range(n_workers):
        total = worker_sizes[w]
        if n_classes == 1:
            # Degenerate case
            only = int(classes[0])
            worker_indices[w].extend(pop_many(pools[only], total))
            continue

        # Majority class selection:
        # - Binary: odd worker -> class 0 majority, even worker -> class 1 majority (matches your worker logic)
        # - Multi-class: worker i majority = classes[i % n_classes]
        if n_classes == 2:
            maj = int(classes[0]) if ((w + 1) % 2 == 1) else int(classes[1])
        else:
            maj = int(classes[w % n_classes])

        n_maj = int(round(total * skew))
        n_rest = total - n_maj

        worker_indices[w].extend(pop_many(pools[maj], n_maj))

        # Distribute remaining quota across non-majority classes
        other_classes = [int(c) for c in classes if int(c) != maj]
        if other_classes:
            per = n_rest // len(other_classes)
            extra = n_rest % len(other_classes)
            for i, c in enumerate(other_classes):
                take = per + (1 if i < extra else 0)
                worker_indices[w].extend(pop_many(pools[c], take))

        # If we couldn't fill due to depleted pools, top up from whatever remains
        missing = total - len(worker_indices[w])
        if missing > 0:
            for c in classes:
                if missing <= 0:
                    break
                got = pop_many(pools[int(c)], missing)
                worker_indices[w].extend(got)
                missing = total - len(worker_indices[w])

        rng.shuffle(worker_indices[w])

    return [np.array(ix, dtype=np.int64) for ix in worker_indices]


def _counts_matrix(y: np.ndarray, partitions: Sequence[np.ndarray], n_classes: int) -> np.ndarray:
    """Matrix shape (n_partitions, n_classes) with class counts."""
    mat = np.zeros((len(partitions), n_classes), dtype=np.int64)
    for i, idx in enumerate(partitions):
        yi = y[idx]
        counts = np.bincount(yi, minlength=n_classes)
        mat[i, :] = counts
    return mat


def _group_workers_into_controllers(worker_counts: np.ndarray, n_controllers: int) -> Tuple[np.ndarray, List[str]]:
    """
    Sum worker distributions into controller distributions.
    Workers are assigned round-robin: worker i -> controller (i % n_controllers).
    """
    if n_controllers <= 0:
        return np.zeros((0, worker_counts.shape[1]), dtype=np.int64), []

    ctrl = np.zeros((n_controllers, worker_counts.shape[1]), dtype=np.int64)
    for w in range(worker_counts.shape[0]):
        c = w % n_controllers
        ctrl[c, :] += worker_counts[w, :]
    ctrl_labels = [f"controller{c+1}" for c in range(n_controllers)]
    return ctrl, ctrl_labels


def _plot_heatmap(
    data: np.ndarray,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    title: str,
    out_path: str,
    normalize: str = "none",
) -> None:
    """
    Plot a heatmap with numeric annotations using matplotlib only (no seaborn/pandas).

    normalize:
      - "none": raw counts
      - "row": row-normalized percentages (0..100)
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if normalize not in ("none", "row"):
        raise ValueError("normalize must be one of: none, row")

    plot_data = data.astype(np.float64)
    fmt = "d"
    cbar_label = "Count"

    if normalize == "row":
        row_sums = plot_data.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        plot_data = (plot_data / row_sums) * 100.0
        fmt = ".1f"
        cbar_label = "Percent (row-normalized)"

    n_rows, n_cols = plot_data.shape
    fig_w = max(7.5, 0.9 * n_cols + 4.0)
    fig_h = max(4.0, 0.45 * n_rows + 2.5)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(plot_data, aspect="auto", cmap="Blues")

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Node", fontsize=12)

    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(col_labels, rotation=25, ha="right")
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(row_labels)

    # Annotate
    for i in range(n_rows):
        for j in range(n_cols):
            val = plot_data[i, j]
            if normalize == "none":
                txt = f"{int(data[i, j])}"
            else:
                txt = f"{val:{fmt}}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9, color="black")

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label(cbar_label, rotation=90)

    ax.grid(False)
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate class-distribution heatmap images (IID / Non-IID)")
    parser.add_argument(
        "--dataset-csv",
        type=str,
        default="dataset_sdn.csv",
        help="Path to CSV dataset (default: dataset_sdn.csv)",
    )
    parser.add_argument(
        "--label-col",
        type=str,
        default="label",
        help="Label column name in CSV (default: label)",
    )
    parser.add_argument("--n-workers", type=int, default=6, help="Number of FL workers/clients (default: 6)")
    parser.add_argument(
        "--n-controllers",
        type=int,
        default=2,
        help="Number of controllers to group workers into (0 disables controller heatmap) (default: 2)",
    )
    parser.add_argument(
        "--partition",
        type=str,
        choices=["iid", "noniid", "both"],
        default="both",
        help="Which partition type(s) to generate (default: both)",
    )
    parser.add_argument(
        "--skew",
        type=float,
        default=0.8,
        help="Non-IID majority fraction (binary: 0.8 means 80/20) (default: 0.8)",
    )
    parser.add_argument(
        "--use-train-split",
        action="store_true",
        default=True,
        help="Use stratified 80%% train split before partitioning (default: True)",
    )
    parser.add_argument(
        "--no-train-split",
        dest="use_train_split",
        action="store_false",
        help="Use the full dataset (no train/test split) before partitioning",
    )
    parser.add_argument("--train-frac", type=float, default=0.8, help="Train fraction if --use-train-split (default: 0.8)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument(
        "--normalize",
        type=str,
        choices=["none", "row"],
        default="none",
        help="Heatmap normalization (default: none). 'row' shows row-normalized percentages.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="visualizations",
        help="Directory to save PNG images (default: visualizations)",
    )

    args = parser.parse_args()

    if args.n_workers <= 0:
        raise ValueError("--n-workers must be >= 1")
    if args.n_controllers < 0:
        raise ValueError("--n-controllers must be >= 0")

    y_all, enc = _read_labels_from_csv(args.dataset_csv, args.label_col)
    n_classes = len(enc.id_to_raw)

    if args.use_train_split:
        train_idx = _train_split_indices(y_all, train_frac=args.train_frac, seed=args.seed)
        y = y_all[train_idx]
    else:
        y = y_all

    class_labels = [f"Class {enc.id_to_raw[i]}" for i in range(n_classes)]
    worker_labels = [f"worker{w+1}" for w in range(args.n_workers)]

    def gen_and_save(kind: str) -> None:
        if kind == "iid":
            parts = _partition_iid(y, n_workers=args.n_workers, seed=args.seed)
            dist_name = "iid"
            dist_title = "IID"
        else:
            parts = _partition_noniid(y, n_workers=args.n_workers, seed=args.seed, skew=args.skew)
            dist_name = "noniid"
            dist_title = "Non-IID (class-skewed)"

        worker_counts = _counts_matrix(y, parts, n_classes=n_classes)

        # Worker heatmap
        out_workers = os.path.join(args.output_dir, f"class_distribution_workers_{dist_name}.png")
        _plot_heatmap(
            worker_counts,
            row_labels=worker_labels,
            col_labels=class_labels,
            title=f"Class Distribution Across Workers ({dist_title})",
            out_path=out_workers,
            normalize=args.normalize,
        )

        # Controller heatmap (optional)
        if args.n_controllers > 0:
            ctrl_counts, ctrl_labels = _group_workers_into_controllers(worker_counts, n_controllers=args.n_controllers)
            out_ctrl = os.path.join(args.output_dir, f"class_distribution_controllers_{dist_name}.png")
            _plot_heatmap(
                ctrl_counts,
                row_labels=ctrl_labels,
                col_labels=class_labels,
                title=f"Class Distribution Across Controllers ({dist_title})",
                out_path=out_ctrl,
                normalize=args.normalize,
            )

            # Combined (controllers + workers) heatmap for "between workers and controllers"
            combined = np.vstack([ctrl_counts, worker_counts])
            combined_labels = list(ctrl_labels) + worker_labels
            out_combined = os.path.join(args.output_dir, f"class_distribution_controllers_workers_{dist_name}.png")
            _plot_heatmap(
                combined,
                row_labels=combined_labels,
                col_labels=class_labels,
                title=f"Class Distribution: Controllers + Workers ({dist_title})",
                out_path=out_combined,
                normalize=args.normalize,
            )

        print(f"✅ Saved heatmaps for {dist_title} into: {os.path.abspath(args.output_dir)}")

    if args.partition in ("iid", "both"):
        gen_and_save("iid")
    if args.partition in ("noniid", "both"):
        gen_and_save("noniid")


if __name__ == "__main__":
    main()


