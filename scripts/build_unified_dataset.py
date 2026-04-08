from __future__ import annotations

import argparse
from io import StringIO
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT_DIR / "combined_sofc_dataset.csv"
EXCLUDED_OUTPUTS = {
    "combined_sofc_dataset.csv",
    "dataset_pareto_front.csv",
    "ga_pareto_front.csv",
}


def resolve_path(path_value: Path | str, base_dir: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def is_model_csv(path: Path) -> bool:
    if path.name in EXCLUDED_OUTPUTS:
        return False

    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            for raw_line in handle:
                stripped = raw_line.strip()
                if not stripped:
                    continue
                return stripped.startswith("#")
    except UnicodeDecodeError:
        return False

    return False


def read_model_csv(path: Path) -> pd.DataFrame:
    comment_lines: list[str] = []
    data_lines: list[str] = []

    with path.open("r", encoding="utf-8-sig") as handle:
        for raw_line in handle:
            if raw_line.lstrip().startswith("#"):
                comment_lines.append(raw_line.strip())
                continue
            if raw_line.strip():
                data_lines.append(raw_line)

    if not data_lines:
        raise ValueError(f"No tabular data found in {path.name}")

    frame = pd.read_csv(StringIO("".join(data_lines)))
    numeric_columns = [column for column in frame.columns if column != "Name"]
    frame[numeric_columns] = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    frame.insert(0, "source_file", path.name)
    frame.insert(1, "source_stem", path.stem)
    frame.insert(2, "run_timestamp", extract_timestamp(comment_lines))
    frame.insert(4, "dp_index", frame["Name"].str.extract(r"(\d+)").astype("Int64"))

    return frame


def extract_timestamp(comment_lines: list[str]) -> str | None:
    for line in comment_lines:
        candidate = line.lstrip("#").strip()
        if "/" in candidate and ":" in candidate:
            return candidate
    return None


def collect_frames(data_dir: Path) -> tuple[list[Path], list[pd.DataFrame]]:
    source_files = sorted(
        (path for path in data_dir.glob("*.csv") if is_model_csv(path)),
        key=lambda item: item.name.lower(),
    )
    if not source_files:
        raise FileNotFoundError(f"No model CSV files were found in {data_dir}")

    frames = [read_model_csv(path) for path in source_files]
    return source_files, frames


def build_dataset(data_dir: Path, output_path: Path) -> pd.DataFrame:
    source_files, frames = collect_frames(data_dir)
    combined = pd.concat(frames, ignore_index=True)
    combined.insert(0, "record_id", range(1, len(combined) + 1))
    combined = combined.sort_values(["source_file", "dp_index"], kind="stable").reset_index(drop=True)
    combined["record_id"] = range(1, len(combined) + 1)
    combined.to_csv(output_path, index=False)

    missing_rows = int(combined[["P62", "P76"]].isna().any(axis=1).sum())

    print(f"Source files merged: {len(source_files)}")
    print(f"Rows written: {len(combined)}")
    print(f"Columns written: {len(combined.columns)}")
    print(f"Rows with missing P62 or P76: {missing_rows}")
    for column in ["P27", "P46", "P47", "P52", "P62", "P76"]:
        if column in combined.columns:
            print(
                f"{column}: min={combined[column].min():.6g}, "
                f"max={combined[column].max():.6g}, unique={combined[column].nunique()}"
            )
    print(f"Unified dataset saved to: {output_path}")
    return combined


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge all SOFC model CSV files into a single dataset."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT_DIR,
        help="Directory that contains the raw model CSV files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output path for the merged dataset CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = resolve_path(args.data_dir, ROOT_DIR)
    output_path = resolve_path(args.output, ROOT_DIR)
    build_dataset(data_dir=data_dir, output_path=output_path)


if __name__ == "__main__":
    main()