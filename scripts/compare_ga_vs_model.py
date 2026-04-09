"""Compare GA-predicted values with actual model results run on GA-found parameters."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
GA_SOLUTION_PATH = ROOT_DIR / "ga_best_solution.csv"
MODEL_RESULT_PATH = ROOT_DIR / "Best of the best point 2.csv"

FEATURE_COLUMNS = ["P27", "P46", "P47", "P52"]
TARGET_COLUMNS = ["P76", "P62", "P79", "P89"]

LABELS = {
    "P27": "cell-voltage [V]",
    "P46": "air-in-ratio",
    "P47": "fuel-flow [ml/min]",
    "P52": "gas-in-t [K]",
    "P76": "current-density [A m^-2]",
    "P62": "total-losses [V]",
    "P79": "fuel-out-h2",
    "P89": "power-efficiency",
}

OUTPUT_CSV = ROOT_DIR / "ga_vs_model_comparison.csv"
OUTPUT_FIGURE = ROOT_DIR / "ga_vs_model_comparison.png"


def load_model_result(path: Path) -> pd.Series:
    """Load the actual model result CSV (COMSOL-style header with comment lines)."""
    df = pd.read_csv(path, comment="#")
    if df.empty:
        raise ValueError(f"No data rows found in {path}")
    return df.iloc[0]


def load_ga_solution(path: Path) -> pd.Series:
    """Load the GA best solution CSV."""
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"No data rows found in {path}")
    return df.iloc[0]


def build_comparison(ga: pd.Series, model: pd.Series) -> pd.DataFrame:
    """Build a comparison table of GA-predicted vs actual model values."""
    rows = []

    # Input parameters
    for col in FEATURE_COLUMNS:
        ga_val = ga[col]
        model_val = model[col]
        abs_err = model_val - ga_val
        rel_err = abs_err / model_val * 100 if model_val != 0 else np.nan
        rows.append({
            "parameter": col,
            "description": LABELS.get(col, col),
            "category": "input",
            "ga_predicted": ga_val,
            "model_actual": model_val,
            "abs_error": abs_err,
            "rel_error_pct": rel_err,
        })

    # Output targets
    ga_target_map = {
        "P76": "predicted_P76",
        "P62": "predicted_P62",
        "P79": "predicted_P79",
        "P89": "predicted_P89",
    }
    for col in TARGET_COLUMNS:
        ga_col = ga_target_map[col]
        ga_val = ga[ga_col]
        model_val = model[col]
        abs_err = model_val - ga_val
        rel_err = abs_err / model_val * 100 if model_val != 0 else np.nan
        rows.append({
            "parameter": col,
            "description": LABELS.get(col, col),
            "category": "output",
            "ga_predicted": ga_val,
            "model_actual": model_val,
            "abs_error": abs_err,
            "rel_error_pct": rel_err,
        })

    return pd.DataFrame(rows)


def plot_comparison(comparison: pd.DataFrame, output_path: Path) -> None:
    """Create a visual comparison of GA predictions vs actual model results."""
    outputs = comparison[comparison["category"] == "output"].copy()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors_ga = "#dc2626"
    colors_model = "#2563eb"

    for ax, (_, row) in zip(axes.flat, outputs.iterrows()):
        param = row["parameter"]
        ga_val = row["ga_predicted"]
        model_val = row["model_actual"]
        rel_err = row["rel_error_pct"]

        bars = ax.bar(
            ["GA (surrogate)", "Model (actual)"],
            [ga_val, model_val],
            color=[colors_ga, colors_model],
            width=0.5,
            edgecolor="#111827",
            linewidth=0.8,
        )

        for bar, val in zip(bars, [ga_val, model_val]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.4f}" if abs(val) < 10 else f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        ax.set_title(f"{param}: {LABELS[param]}\n(rel. error: {rel_err:+.1f}%)", fontsize=11)
        ax.set_ylabel(LABELS[param])
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "GA surrogate prediction vs actual model result",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ga = load_ga_solution(GA_SOLUTION_PATH)
    model = load_model_result(MODEL_RESULT_PATH)

    comparison = build_comparison(ga, model)
    comparison.to_csv(OUTPUT_CSV, index=False)

    print("=" * 70)
    print("  GA surrogate prediction vs actual model result")
    print("=" * 70)

    print("\n--- Input parameters ---")
    inputs = comparison[comparison["category"] == "input"]
    for _, row in inputs.iterrows():
        print(
            f"  {row['parameter']:>4s} ({row['description']:<22s}): "
            f"GA={row['ga_predicted']:>12.4f}  Model={row['model_actual']:>12.4f}  "
            f"err={row['rel_error_pct']:>+7.2f}%"
        )

    print("\n--- Output targets ---")
    outputs = comparison[comparison["category"] == "output"]
    for _, row in outputs.iterrows():
        print(
            f"  {row['parameter']:>4s} ({row['description']:<25s}): "
            f"GA={row['ga_predicted']:>12.4f}  Model={row['model_actual']:>12.4f}  "
            f"err={row['rel_error_pct']:>+7.2f}%"
        )

    plot_comparison(comparison, OUTPUT_FIGURE)

    print(f"\nComparison table saved to: {OUTPUT_CSV}")
    print(f"Comparison plot saved to:  {OUTPUT_FIGURE}")


if __name__ == "__main__":
    main()
