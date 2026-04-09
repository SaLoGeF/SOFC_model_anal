from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "axes.linewidth": 0.8,
})


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT_DIR / "combined_sofc_dataset.csv"
DEFAULT_OUTPUT = ROOT_DIR / "correlation_heatmap.png"


def resolve_path(path_value: Path | str, base_dir: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def select_columns(frame: pd.DataFrame, columns: list[str] | None) -> pd.DataFrame:
    numeric = frame.select_dtypes(include="number").drop(columns=["record_id", "dp_index"], errors="ignore")
    if columns:
        missing = [column for column in columns if column not in numeric.columns]
        if missing:
            raise KeyError(f"Columns not found in dataset: {', '.join(missing)}")
        return numeric[columns]
    return numeric


def print_top_correlations(correlation: pd.DataFrame, target: str, top_n: int = 8) -> None:
    if target not in correlation.columns:
        return
    ranked = correlation[target].drop(labels=[target], errors="ignore")
    ranked = ranked.reindex(ranked.abs().sort_values(ascending=False).index).head(top_n)
    print(f"Top correlations for {target}:")
    print(ranked.to_string())
    print()


def build_heatmap(dataset_path: Path, output_path: Path, columns: list[str] | None, method: Literal["pearson", "kendall", "spearman"]) -> None:
    frame = pd.read_csv(dataset_path)
    selected = select_columns(frame, columns)
    correlation = selected.corr(method=method)

    if correlation.empty:
        raise ValueError("No numeric columns are available for the correlation map.")

    figure_scale = max(10, min(22, int(len(correlation.columns) * 0.6)))
    annotate = len(correlation.columns) <= 12
    mask = np.triu(np.ones_like(correlation, dtype=bool))

    sns.set_theme(style="white")
    plt.figure(figsize=(figure_scale, figure_scale))
    sns.heatmap(
        correlation,
        mask=mask,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.3,
        annot=annotate,
        fmt=".2f",
        cbar_kws={"shrink": 0.8},
    )
    plt.title(f"Корреляционная матрица ТОЭ ({method})")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Heatmap saved to: {output_path}")
    print_top_correlations(correlation, "P76")
    print_top_correlations(correlation, "P62")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a correlation heatmap for the merged SOFC dataset.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to the merged dataset CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output image path for the heatmap.",
    )
    parser.add_argument(
        "--method",
        choices=["pearson", "spearman", "kendall"],
        default="pearson",
        help="Correlation method.",
    )
    parser.add_argument(
        "--columns",
        nargs="*",
        default=None,
        help="Optional list of numeric columns to include. By default the script uses all numeric columns.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = resolve_path(args.dataset, ROOT_DIR)
    output_path = resolve_path(args.output, ROOT_DIR)
    build_heatmap(dataset_path=dataset_path, output_path=output_path, columns=args.columns, method=args.method)


if __name__ == "__main__":
    main()