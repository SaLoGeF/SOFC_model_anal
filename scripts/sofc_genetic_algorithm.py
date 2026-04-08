from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT_DIR / "combined_sofc_dataset.csv"
DEFAULT_OUTPUT = ROOT_DIR / "ga_best_solution.csv"
DEFAULT_HISTORY_OUTPUT = ROOT_DIR / "ga_convergence_history.csv"
DEFAULT_CONVERGENCE_FIGURE = ROOT_DIR / "ga_convergence.png"
DEFAULT_ACCURACY_FIGURE = ROOT_DIR / "ga_model_accuracy.png"
FEATURE_COLUMNS = ["P27", "P46", "P47", "P52"]
CURRENT_TARGET = "P76"
LOSS_TARGET = "P62"


@dataclass
class Individual:
    genes: np.ndarray
    predicted_current: float
    predicted_losses: float
    rank: int = 0
    crowding: float = 0.0

    def clone(self) -> "Individual":
        return Individual(
            genes=self.genes.copy(),
            predicted_current=self.predicted_current,
            predicted_losses=self.predicted_losses,
            rank=self.rank,
            crowding=self.crowding,
        )


def resolve_path(path_value: Path | str, base_dir: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def dominates(left: Individual, right: Individual) -> bool:
    no_worse = (
        left.predicted_current >= right.predicted_current
        and left.predicted_losses <= right.predicted_losses
    )
    strictly_better = (
        left.predicted_current > right.predicted_current
        or left.predicted_losses < right.predicted_losses
    )
    return no_worse and strictly_better


def fast_non_dominated_sort(population: list[Individual]) -> list[list[Individual]]:
    domination_counts = [0] * len(population)
    dominated_sets: list[list[int]] = [[] for _ in population]
    fronts: list[list[int]] = [[]]

    for i, left in enumerate(population):
        for j, right in enumerate(population):
            if i == j:
                continue
            if dominates(left, right):
                dominated_sets[i].append(j)
            elif dominates(right, left):
                domination_counts[i] += 1
        if domination_counts[i] == 0:
            left.rank = 0
            fronts[0].append(i)

    current_front = 0
    while current_front < len(fronts) and fronts[current_front]:
        next_front: list[int] = []
        for index in fronts[current_front]:
            for dominated_index in dominated_sets[index]:
                domination_counts[dominated_index] -= 1
                if domination_counts[dominated_index] == 0:
                    population[dominated_index].rank = current_front + 1
                    next_front.append(dominated_index)
        if next_front:
            fronts.append(next_front)
        current_front += 1

    return [[population[index] for index in front] for front in fronts if front]


def assign_crowding_distance(front: list[Individual]) -> None:
    if not front:
        return
    if len(front) <= 2:
        for individual in front:
            individual.crowding = float("inf")
        return

    for individual in front:
        individual.crowding = 0.0

    objectives = ["predicted_current", "predicted_losses"]
    for objective in objectives:
        front.sort(key=lambda individual: getattr(individual, objective))
        front[0].crowding = float("inf")
        front[-1].crowding = float("inf")
        min_value = getattr(front[0], objective)
        max_value = getattr(front[-1], objective)
        scale = max_value - min_value
        if scale == 0:
            continue
        for index in range(1, len(front) - 1):
            prev_value = getattr(front[index - 1], objective)
            next_value = getattr(front[index + 1], objective)
            front[index].crowding += (next_value - prev_value) / scale


def tournament_select(population: list[Individual], rng: np.random.Generator) -> Individual:
    first, second = rng.choice(population, size=2, replace=False)
    if first.rank != second.rank:
        return first if first.rank < second.rank else second
    if first.crowding != second.crowding:
        return first if first.crowding > second.crowding else second
    return first if rng.random() < 0.5 else second


def clamp_genes(genes: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    lower = bounds[:, 0]
    upper = bounds[:, 1]
    return np.clip(genes, lower, upper)


def crossover(
    left: Individual,
    right: Individual,
    bounds: np.ndarray,
    rng: np.random.Generator,
    crossover_rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    if rng.random() > crossover_rate:
        return left.genes.copy(), right.genes.copy()

    mix = rng.random(len(bounds))
    child_a = mix * left.genes + (1.0 - mix) * right.genes
    child_b = mix * right.genes + (1.0 - mix) * left.genes
    return clamp_genes(child_a, bounds), clamp_genes(child_b, bounds)


def mutate(
    genes: np.ndarray,
    bounds: np.ndarray,
    rng: np.random.Generator,
    mutation_rate: float,
    mutation_scale: float,
) -> np.ndarray:
    mutated = genes.copy()
    spans = bounds[:, 1] - bounds[:, 0]
    for index, span in enumerate(spans):
        if rng.random() < mutation_rate:
            mutated[index] += rng.normal(0.0, span * mutation_scale)
    return clamp_genes(mutated, bounds)


def evaluate_genes(
    genes: np.ndarray,
    current_model: RandomForestRegressor,
    loss_model: RandomForestRegressor,
) -> Individual:
    current_prediction = float(current_model.predict(genes.reshape(1, -1))[0])
    loss_prediction = float(loss_model.predict(genes.reshape(1, -1))[0])
    return Individual(
        genes=genes,
        predicted_current=current_prediction,
        predicted_losses=loss_prediction,
    )


def initialize_population(
    population_size: int,
    bounds: np.ndarray,
    seed_points: np.ndarray,
    rng: np.random.Generator,
    current_model: RandomForestRegressor,
    loss_model: RandomForestRegressor,
) -> list[Individual]:
    population: list[Individual] = []
    shuffled_indices = rng.permutation(len(seed_points))
    seed_count = min(len(seed_points), population_size // 2)

    for index in shuffled_indices[:seed_count]:
        population.append(evaluate_genes(seed_points[index].copy(), current_model, loss_model))

    while len(population) < population_size:
        genes = rng.uniform(bounds[:, 0], bounds[:, 1])
        population.append(evaluate_genes(genes, current_model, loss_model))

    return population


def evolve_population(
    current_model: RandomForestRegressor,
    loss_model: RandomForestRegressor,
    bounds: np.ndarray,
    seed_points: np.ndarray,
    population_size: int,
    generations: int,
    crossover_rate: float,
    mutation_rate: float,
    mutation_scale: float,
    rng: np.random.Generator,
) -> tuple[list[Individual], pd.DataFrame, Individual]:
    population = initialize_population(
        population_size=population_size,
        bounds=bounds,
        seed_points=seed_points,
        rng=rng,
        current_model=current_model,
        loss_model=loss_model,
    )

    fronts = fast_non_dominated_sort(population)
    for front in fronts:
        assign_crowding_distance(front)
    best_candidate, initial_summary = extract_compromise_candidate(fronts[0])
    best_distance_so_far = initial_summary["best_distance_to_ideal"]
    history_rows = [
        {
            "generation": 0,
            **initial_summary,
            "best_so_far_distance_to_ideal": best_distance_so_far,
            "best_so_far_P76": best_candidate.predicted_current,
            "best_so_far_P62": best_candidate.predicted_losses,
        }
    ]

    progress_interval = max(1, generations // 5)

    for generation in range(1, generations + 1):
        fronts = fast_non_dominated_sort(population)
        for front in fronts:
            assign_crowding_distance(front)

        offspring: list[Individual] = []
        while len(offspring) < population_size:
            parent_a = tournament_select(population, rng)
            parent_b = tournament_select(population, rng)
            child_a_genes, child_b_genes = crossover(
                parent_a,
                parent_b,
                bounds=bounds,
                rng=rng,
                crossover_rate=crossover_rate,
            )
            child_a_genes = mutate(
                child_a_genes,
                bounds=bounds,
                rng=rng,
                mutation_rate=mutation_rate,
                mutation_scale=mutation_scale,
            )
            child_b_genes = mutate(
                child_b_genes,
                bounds=bounds,
                rng=rng,
                mutation_rate=mutation_rate,
                mutation_scale=mutation_scale,
            )
            offspring.append(evaluate_genes(child_a_genes, current_model, loss_model))
            if len(offspring) < population_size:
                offspring.append(evaluate_genes(child_b_genes, current_model, loss_model))

        combined = population + offspring
        combined_fronts = fast_non_dominated_sort(combined)
        next_population: list[Individual] = []
        for front in combined_fronts:
            assign_crowding_distance(front)
            if len(next_population) + len(front) <= population_size:
                next_population.extend(individual.clone() for individual in front)
                continue
            ordered = sorted(front, key=lambda individual: individual.crowding, reverse=True)
            remaining = population_size - len(next_population)
            next_population.extend(individual.clone() for individual in ordered[:remaining])
            break

        population = next_population

        fronts = fast_non_dominated_sort(population)
        for front in fronts:
            assign_crowding_distance(front)

        current_candidate, front_summary = extract_compromise_candidate(fronts[0])
        if front_summary["best_distance_to_ideal"] < best_distance_so_far:
            best_candidate = current_candidate
            best_distance_so_far = front_summary["best_distance_to_ideal"]

        summary = {
            "generation": generation,
            **front_summary,
            "best_so_far_distance_to_ideal": best_distance_so_far,
            "best_so_far_P76": best_candidate.predicted_current,
            "best_so_far_P62": best_candidate.predicted_losses,
        }
        history_rows.append(summary)

        if generation == 1 or generation % progress_interval == 0 or generation == generations:
            print(
                f"Generation {generation:>3}: front_size={int(summary['front_size'])}, "
                f"current_distance={summary['best_distance_to_ideal']:.4f}, "
                f"best_so_far_distance={summary['best_so_far_distance_to_ideal']:.4f}, "
                f"best_so_far_P76={summary['best_so_far_P76']:.4f}, "
                f"best_so_far_P62={summary['best_so_far_P62']:.4f}"
            )

    final_front = fronts[0]
    assign_crowding_distance(final_front)
    history = pd.DataFrame(history_rows)
    return final_front, history, best_candidate


def train_surrogate_models(
    dataset: pd.DataFrame,
    feature_columns: list[str],
    random_state: int,
) -> tuple[RandomForestRegressor, RandomForestRegressor, dict[str, float], pd.DataFrame]:
    features = dataset[feature_columns].to_numpy(dtype=float)
    current_target = dataset[CURRENT_TARGET].to_numpy(dtype=float)
    loss_target = dataset[LOSS_TARGET].to_numpy(dtype=float)

    (
        x_train,
        x_test,
        current_train,
        current_test,
        loss_train,
        loss_test,
    ) = train_test_split(
        features,
        current_target,
        loss_target,
        test_size=0.2,
        random_state=random_state,
    )

    model_kwargs = {
        "n_estimators": 400,
        "random_state": random_state,
        "n_jobs": -1,
        "min_samples_leaf": 1,
    }

    current_model = RandomForestRegressor(**model_kwargs)
    loss_model = RandomForestRegressor(**model_kwargs)
    current_model.fit(x_train, current_train)
    loss_model.fit(x_train, loss_train)

    current_pred = current_model.predict(x_test)
    loss_pred = loss_model.predict(x_test)
    metrics = {
        "current_r2": r2_score(current_test, current_pred),
        "current_mae": mean_absolute_error(current_test, current_pred),
        "loss_r2": r2_score(loss_test, loss_pred),
        "loss_mae": mean_absolute_error(loss_test, loss_pred),
    }
    accuracy_frame = pd.DataFrame(
        {
            "actual_P76": current_test,
            "predicted_P76": current_pred,
            "actual_P62": loss_test,
            "predicted_P62": loss_pred,
        }
    )

    current_model.fit(features, current_target)
    loss_model.fit(features, loss_target)
    return current_model, loss_model, metrics, accuracy_frame


def dataset_pareto_front(dataset: pd.DataFrame) -> pd.DataFrame:
    candidates = dataset.reset_index(drop=True)
    pareto_mask = np.ones(len(candidates), dtype=bool)
    current_values = candidates[CURRENT_TARGET].to_numpy(dtype=float)
    loss_values = candidates[LOSS_TARGET].to_numpy(dtype=float)

    for index in range(len(candidates)):
        dominating = (
            (current_values >= current_values[index])
            & (loss_values <= loss_values[index])
            & (
                (current_values > current_values[index])
                | (loss_values < loss_values[index])
            )
        )
        dominating[index] = False
        if dominating.any():
            pareto_mask[index] = False

    front = candidates.loc[pareto_mask].copy()
    return front.sort_values([LOSS_TARGET, CURRENT_TARGET], ascending=[True, False]).reset_index(drop=True)


def normalized_distance_to_ideal(current_values: np.ndarray, loss_values: np.ndarray) -> np.ndarray:
    current_span = current_values.max() - current_values.min()
    loss_span = loss_values.max() - loss_values.min()
    current_span = current_span if current_span else 1.0
    loss_span = loss_span if loss_span else 1.0

    current_score = (current_values - current_values.min()) / current_span
    loss_score = (loss_values.max() - loss_values) / loss_span
    return np.sqrt((1.0 - current_score) ** 2 + (1.0 - loss_score) ** 2)


def extract_compromise_candidate(front: list[Individual]) -> tuple[Individual, dict[str, float]]:
    current_values = np.array([individual.predicted_current for individual in front], dtype=float)
    loss_values = np.array([individual.predicted_losses for individual in front], dtype=float)
    distances = normalized_distance_to_ideal(current_values, loss_values)
    best_index = int(np.argmin(distances))

    summary = {
        "front_size": float(len(front)),
        "best_distance_to_ideal": float(distances[best_index]),
        "best_predicted_P76": float(current_values[best_index]),
        "best_predicted_P62": float(loss_values[best_index]),
        "front_max_P76": float(current_values.max()),
        "front_min_P62": float(loss_values.min()),
    }
    return front[best_index].clone(), summary


def attach_nearest_dataset_point(
    front: pd.DataFrame,
    dataset: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    feature_matrix = dataset[feature_columns].to_numpy(dtype=float)
    scales = np.ptp(feature_matrix, axis=0)
    scales[scales == 0] = 1.0

    nearest_indices: list[int] = []
    nearest_distances: list[float] = []
    for _, row in front.iterrows():
        candidate = row[feature_columns].to_numpy(dtype=float)
        distances = np.linalg.norm((feature_matrix - candidate) / scales, axis=1)
        nearest_index = int(np.argmin(distances))
        nearest_indices.append(nearest_index)
        nearest_distances.append(float(distances[nearest_index]))

    nearest_rows = dataset.iloc[nearest_indices].reset_index(drop=True)
    enriched = front.reset_index(drop=True).copy()
    enriched["nearest_source_file"] = nearest_rows["source_file"]
    enriched["nearest_name"] = nearest_rows["Name"]
    enriched["nearest_dataset_P76"] = nearest_rows[CURRENT_TARGET]
    enriched["nearest_dataset_P62"] = nearest_rows[LOSS_TARGET]
    enriched["feature_distance_to_dataset"] = nearest_distances
    return enriched


def individuals_to_frame(front: list[Individual], dataset: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    data = []
    for individual in front:
        row = {column: value for column, value in zip(feature_columns, individual.genes)}
        row["predicted_P76"] = individual.predicted_current
        row["predicted_P62"] = individual.predicted_losses
        row["rank"] = individual.rank
        row["crowding"] = individual.crowding
        data.append(row)

    frame = pd.DataFrame(data).drop_duplicates(subset=feature_columns + ["predicted_P76", "predicted_P62"])
    frame["distance_to_ideal"] = normalized_distance_to_ideal(
        frame["predicted_P76"].to_numpy(dtype=float),
        frame["predicted_P62"].to_numpy(dtype=float),
    )
    frame = frame.sort_values(
        ["distance_to_ideal", "predicted_P62", "predicted_P76"],
        ascending=[True, True, False],
    ).reset_index(drop=True)
    return attach_nearest_dataset_point(frame, dataset=dataset, feature_columns=feature_columns)


def individual_to_frame(individual: Individual, dataset: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            {
                **{column: value for column, value in zip(feature_columns, individual.genes)},
                "predicted_P76": individual.predicted_current,
                "predicted_P62": individual.predicted_losses,
                "rank": individual.rank,
                "crowding": individual.crowding,
            }
        ]
    )
    return attach_nearest_dataset_point(frame, dataset=dataset, feature_columns=feature_columns)


def build_best_solution_frame(
    solution_frame: pd.DataFrame,
    best_distance: float,
    metrics: dict[str, float],
    population_size: int,
    generations: int,
) -> pd.DataFrame:
    best_solution = solution_frame.iloc[[0]].copy()
    best_solution.insert(0, "selection_rule", "min_distance_to_ideal_over_all_generations")
    best_solution["distance_to_ideal"] = best_distance
    best_solution["population_size"] = population_size
    best_solution["generations"] = generations
    best_solution["surrogate_P76_R2"] = metrics["current_r2"]
    best_solution["surrogate_P76_MAE"] = metrics["current_mae"]
    best_solution["surrogate_P62_R2"] = metrics["loss_r2"]
    best_solution["surrogate_P62_MAE"] = metrics["loss_mae"]
    return best_solution


def plot_convergence(history: pd.DataFrame, output_path: Path) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    axes[0].plot(history["generation"], history["best_so_far_distance_to_ideal"], marker="o", color="#111827", linewidth=2)
    axes[0].set_title("GA convergence")
    axes[0].set_ylabel("Distance to ideal")
    axes[0].grid(alpha=0.3)

    axes[1].plot(history["generation"], history["front_max_P76"], marker="o", color="#2563eb", label="Front max P76")
    axes[1].plot(history["generation"], history["best_so_far_P76"], marker="s", color="#0f766e", label="Best-so-far compromise P76")
    axes[1].set_ylabel("P76 [A m^-2]")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(history["generation"], history["front_min_P62"], marker="o", color="#dc2626", label="Front min P62")
    axes[2].plot(history["generation"], history["best_so_far_P62"], marker="s", color="#7c2d12", label="Best-so-far compromise P62")
    axes[2].set_xlabel("Generation")
    axes[2].set_ylabel("P62 [V]")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_model_accuracy(accuracy_frame: pd.DataFrame, metrics: dict[str, float], output_path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14, 6))
    plot_specs = [
        ("actual_P76", "predicted_P76", "P76 current-density", metrics["current_r2"], metrics["current_mae"], "#2563eb"),
        ("actual_P62", "predicted_P62", "P62 total-losses", metrics["loss_r2"], metrics["loss_mae"], "#dc2626"),
    ]

    for axis, (actual_column, predicted_column, title, r2_value, mae_value, color) in zip(axes, plot_specs):
        actual = accuracy_frame[actual_column].to_numpy(dtype=float)
        predicted = accuracy_frame[predicted_column].to_numpy(dtype=float)
        low = min(actual.min(), predicted.min())
        high = max(actual.max(), predicted.max())

        axis.scatter(actual, predicted, s=50, alpha=0.75, color=color)
        axis.plot([low, high], [low, high], linestyle="--", color="#111827", linewidth=1.5)
        axis.set_title(f"{title}\nR2={r2_value:.4f}, MAE={mae_value:.4f}")
        axis.set_xlabel("Actual")
        axis.set_ylabel("Predicted")
        axis.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_fronts(dataset: pd.DataFrame, dataset_front: pd.DataFrame, ga_front: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(10, 7))
    plt.scatter(
        dataset[LOSS_TARGET],
        dataset[CURRENT_TARGET],
        alpha=0.2,
        s=35,
        label="All dataset points",
        color="#9ca3af",
    )
    plt.scatter(
        dataset_front[LOSS_TARGET],
        dataset_front[CURRENT_TARGET],
        s=70,
        label="Dataset Pareto front",
        color="#2563eb",
    )
    plt.scatter(
        ga_front["predicted_P62"],
        ga_front["predicted_P76"],
        s=70,
        label="GA Pareto front",
        color="#dc2626",
    )

    if not ga_front.empty:
        best = ga_front.iloc[0]
        plt.scatter(best["predicted_P62"], best["predicted_P76"], s=140, color="#111827", label="GA best compromise")

    plt.xlabel("P62 total-losses [V]")
    plt.ylabel("P76 current-density [A m^-2]")
    plt.title("SOFC Pareto front: raw data vs surrogate-assisted GA")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a multi-objective genetic algorithm and export one best predicted SOFC solution."
    )
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
        help="Output CSV path for the single best GA-predicted solution.",
    )
    parser.add_argument(
        "--history-output",
        type=Path,
        default=DEFAULT_HISTORY_OUTPUT,
        help="Output CSV path for GA convergence history.",
    )
    parser.add_argument(
        "--convergence-figure",
        type=Path,
        default=DEFAULT_CONVERGENCE_FIGURE,
        help="Output image path for the GA convergence plot.",
    )
    parser.add_argument(
        "--accuracy-figure",
        type=Path,
        default=DEFAULT_ACCURACY_FIGURE,
        help="Output image path for the surrogate model accuracy plot.",
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=40,
        help="GA population size. Increase it for a deeper search.",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=20,
        help="Number of GA generations. Increase it for a deeper search.",
    )
    parser.add_argument("--crossover-rate", type=float, default=0.9, help="Crossover rate.")
    parser.add_argument("--mutation-rate", type=float, default=0.25, help="Per-gene mutation rate.")
    parser.add_argument(
        "--mutation-scale",
        type=float,
        default=0.08,
        help="Mutation sigma as a fraction of each feature range.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = resolve_path(args.dataset, ROOT_DIR)
    output_path = resolve_path(args.output, ROOT_DIR)
    history_output_path = resolve_path(args.history_output, ROOT_DIR)
    convergence_figure_path = resolve_path(args.convergence_figure, ROOT_DIR)
    accuracy_figure_path = resolve_path(args.accuracy_figure, ROOT_DIR)

    dataset = pd.read_csv(dataset_path)
    required_columns = FEATURE_COLUMNS + [CURRENT_TARGET, LOSS_TARGET, "source_file", "Name"]
    missing_columns = [column for column in required_columns if column not in dataset.columns]
    if missing_columns:
        raise KeyError(f"The dataset is missing required columns: {', '.join(missing_columns)}")

    modeling_dataset = dataset.dropna(subset=FEATURE_COLUMNS + [CURRENT_TARGET, LOSS_TARGET]).copy()
    dropped_rows = len(dataset) - len(modeling_dataset)
    if modeling_dataset.empty:
        raise ValueError("No complete rows are available for GA training and optimization.")
    if dropped_rows:
        print(f"Rows excluded from modeling because of missing values: {dropped_rows}")

    current_model, loss_model, metrics, accuracy_frame = train_surrogate_models(
        dataset=modeling_dataset,
        feature_columns=FEATURE_COLUMNS,
        random_state=args.seed,
    )

    print("Surrogate model validation metrics:")
    print(f"  P76 R2  = {metrics['current_r2']:.4f}")
    print(f"  P76 MAE = {metrics['current_mae']:.4f}")
    print(f"  P62 R2  = {metrics['loss_r2']:.4f}")
    print(f"  P62 MAE = {metrics['loss_mae']:.4f}")
    print("Decision variables used by the GA: P27, P46, P47, P52")

    feature_space = modeling_dataset[FEATURE_COLUMNS].to_numpy(dtype=float)
    bounds = np.column_stack((feature_space.min(axis=0), feature_space.max(axis=0)))
    rng = np.random.default_rng(args.seed)

    front, convergence_history, best_candidate = evolve_population(
        current_model=current_model,
        loss_model=loss_model,
        bounds=bounds,
        seed_points=feature_space,
        population_size=args.population_size,
        generations=args.generations,
        crossover_rate=args.crossover_rate,
        mutation_rate=args.mutation_rate,
        mutation_scale=args.mutation_scale,
        rng=rng,
    )

    ga_front = individuals_to_frame(front, dataset=modeling_dataset, feature_columns=FEATURE_COLUMNS)
    if ga_front.empty:
        raise ValueError("The GA did not produce any candidate solutions.")

    best_candidate_frame = individual_to_frame(best_candidate, dataset=modeling_dataset, feature_columns=FEATURE_COLUMNS)
    best_solution = build_best_solution_frame(
        solution_frame=best_candidate_frame,
        best_distance=float(convergence_history["best_so_far_distance_to_ideal"].min()),
        metrics=metrics,
        population_size=args.population_size,
        generations=args.generations,
    )
    best_solution.to_csv(output_path, index=False)
    convergence_history.to_csv(history_output_path, index=False)

    plot_convergence(convergence_history, convergence_figure_path)
    plot_model_accuracy(accuracy_frame, metrics, accuracy_figure_path)

    print(f"Best GA solution saved to: {output_path}")
    print(f"GA convergence history saved to: {history_output_path}")
    print(f"GA convergence plot saved to: {convergence_figure_path}")
    print(f"GA model accuracy plot saved to: {accuracy_figure_path}")
    print("Best predicted GA solution:")
    print(best_solution.iloc[0].to_string())


if __name__ == "__main__":
    main()