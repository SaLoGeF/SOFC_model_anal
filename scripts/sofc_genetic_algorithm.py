from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


ROOT_DIR = Path(__file__).resolve().parents[1]

# ===========================================================================
#                       НАСТРОЙКИ — ВСЁ В ОДНОМ МЕСТЕ
# ===========================================================================

# --- Пути по умолчанию ------------------------------------------------------
DEFAULT_DATASET = ROOT_DIR / "combined_sofc_dataset.csv"
DEFAULT_OUTPUT = ROOT_DIR / "ga_best_solution.csv"
DEFAULT_HISTORY_OUTPUT = ROOT_DIR / "ga_convergence_history.csv"
DEFAULT_CONVERGENCE_FIGURE = ROOT_DIR / "ga_convergence.png"
DEFAULT_ACCURACY_FIGURE = ROOT_DIR / "ga_model_accuracy.png"

# --- Входные / целевые переменные -------------------------------------------
FEATURE_COLUMNS = ["P27", "P46", "P47", "P52"]

CURRENT_TARGET = "P76"     # Плотность тока (maximize)
LOSS_TARGET = "P62"        # Суммарные потери (minimize)
FUEL_OUT_TARGET = "P79"    # Выход H2 на выходе (minimize)
EFFICIENCY_TARGET = "P89"  # КПД (maximize)
ALL_TARGETS = [CURRENT_TARGET, LOSS_TARGET, FUEL_OUT_TARGET, EFFICIENCY_TARGET]

# --- Параметры генетического алгоритма --------------------------------------
GA_POPULATION_SIZE = 40
GA_GENERATIONS = 20
GA_CROSSOVER_RATE = 0.9
GA_MUTATION_RATE = 0.25       # вероятность мутации каждого гена
GA_MUTATION_SCALE = 0.08      # σ мутации как доля диапазона признака
GA_SEED = 42

# --- Параметры суррогатных ML-моделей ---------------------------------------
ML_TEST_SIZE = 0.2            # доля тестовой выборки

# P76: ансамбль RF + GBR (VotingRegressor)
ML_P76_RF_N_ESTIMATORS = 1000
ML_P76_RF_MIN_SAMPLES_LEAF = 3
ML_P76_GBR_N_ESTIMATORS = 1000
ML_P76_GBR_MAX_DEPTH = 6
ML_P76_GBR_LEARNING_RATE = 0.05
ML_P76_GBR_SUBSAMPLE = 0.8
ML_P76_VOTING_WEIGHTS = [2, 1]  # [RF, GBR]

# P62, P79, P89: RandomForest
ML_RF_N_ESTIMATORS = 400
ML_RF_MIN_SAMPLES_LEAF = 1

# ---------------------------------------------------------------------------
# Fitness-функция (взвешенное расстояние до идеальной точки):
#
#   fitness(x) = sqrt( w_P76 * (1 - s_P76)^2
#                    + w_P62 * (1 - s_P62)^2
#                    # + w_P79 * (1 - s_P79)^2   ← отключён (вес = 0)
#                    + w_P89 * (1 - s_P89)^2 )
#
# где нормализованные score-ы:
#   s_P76 = (P76 - P76_min) / (P76_max - P76_min)   # maximize → чем больше, тем лучше
#   s_P62 = (P62_max - P62) / (P62_max - P62_min)   # minimize → инвертировано
#   s_P79 = (P79_max - P79) / (P79_max - P79_min)   # minimize → отключён (вес = 0)
#   s_P89 = (P89 - P89_min) / (P89_max - P89_min)   # maximize → чем больше, тем лучше
#
# Идеальная точка: s_i = 1 для всех целей.
# Меньшее значение fitness → лучшее решение (ближе к идеалу).
# ---------------------------------------------------------------------------
OBJECTIVE_WEIGHTS = {
    CURRENT_TARGET: 0.4,   # основной
    LOSS_TARGET: 0.2,      # вторичный
    FUEL_OUT_TARGET: 0.0,  # отключён
    EFFICIENCY_TARGET: 0.3,  # основной
}

# ===========================================================================


@dataclass
class Individual:
    genes: np.ndarray
    predicted_current: float
    predicted_losses: float
    predicted_fuel_out: float
    predicted_efficiency: float
    rank: int = 0
    crowding: float = 0.0

    def clone(self) -> "Individual":
        return Individual(
            genes=self.genes.copy(),
            predicted_current=self.predicted_current,
            predicted_losses=self.predicted_losses,
            predicted_fuel_out=self.predicted_fuel_out,
            predicted_efficiency=self.predicted_efficiency,
            rank=self.rank,
            crowding=self.crowding,
        )


def resolve_path(path_value: Path | str, base_dir: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def dominates(left: Individual, right: Individual) -> bool:
    # P76 maximize, P62 minimize, P79 minimize, P89 maximize
    no_worse = (
        left.predicted_current >= right.predicted_current
        and left.predicted_losses <= right.predicted_losses
        and left.predicted_fuel_out <= right.predicted_fuel_out
        and left.predicted_efficiency >= right.predicted_efficiency
    )
    strictly_better = (
        left.predicted_current > right.predicted_current
        or left.predicted_losses < right.predicted_losses
        or left.predicted_fuel_out < right.predicted_fuel_out
        or left.predicted_efficiency > right.predicted_efficiency
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

    objectives = ["predicted_current", "predicted_losses", "predicted_fuel_out", "predicted_efficiency"]
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
    indices = rng.choice(len(population), size=2, replace=False)
    first, second = population[indices[0]], population[indices[1]]
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
    current_model: VotingRegressor,
    loss_model: RandomForestRegressor,
    fuel_out_model: RandomForestRegressor,
    efficiency_model: RandomForestRegressor,
) -> Individual:
    row = genes.reshape(1, -1)
    return Individual(
        genes=genes,
        predicted_current=float(current_model.predict(row)[0]),
        predicted_losses=float(loss_model.predict(row)[0]),
        predicted_fuel_out=float(fuel_out_model.predict(row)[0]),
        predicted_efficiency=float(efficiency_model.predict(row)[0]),
    )


def initialize_population(
    population_size: int,
    bounds: np.ndarray,
    seed_points: np.ndarray,
    rng: np.random.Generator,
    current_model: VotingRegressor,
    loss_model: RandomForestRegressor,
    fuel_out_model: RandomForestRegressor,
    efficiency_model: RandomForestRegressor,
) -> list[Individual]:
    population: list[Individual] = []
    shuffled_indices = rng.permutation(len(seed_points))
    seed_count = min(len(seed_points), population_size // 2)

    for index in shuffled_indices[:seed_count]:
        population.append(evaluate_genes(seed_points[index].copy(), current_model, loss_model, fuel_out_model, efficiency_model))

    while len(population) < population_size:
        genes = rng.uniform(bounds[:, 0], bounds[:, 1])
        population.append(evaluate_genes(genes, current_model, loss_model, fuel_out_model, efficiency_model))

    return population


def evolve_population(
    current_model: VotingRegressor,
    loss_model: RandomForestRegressor,
    fuel_out_model: RandomForestRegressor,
    efficiency_model: RandomForestRegressor,
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
        fuel_out_model=fuel_out_model,
        efficiency_model=efficiency_model,
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
            "best_so_far_P79": best_candidate.predicted_fuel_out,
            "best_so_far_P89": best_candidate.predicted_efficiency,
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
            offspring.append(evaluate_genes(child_a_genes, current_model, loss_model, fuel_out_model, efficiency_model))
            if len(offspring) < population_size:
                offspring.append(evaluate_genes(child_b_genes, current_model, loss_model, fuel_out_model, efficiency_model))

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
            "best_so_far_P79": best_candidate.predicted_fuel_out,
            "best_so_far_P89": best_candidate.predicted_efficiency,
        }
        history_rows.append(summary)

        if generation == 1 or generation % progress_interval == 0 or generation == generations:
            print(
                f"Generation {generation:>3}: front_size={int(summary['front_size'])}, "
                f"distance={summary['best_distance_to_ideal']:.4f}, "
                f"best_dist={summary['best_so_far_distance_to_ideal']:.4f}, "
                f"P76={summary['best_so_far_P76']:.4f}, "
                f"P62={summary['best_so_far_P62']:.4f}, "
                f"P79={summary['best_so_far_P79']:.4f}, "
                f"P89={summary['best_so_far_P89']:.4f}"
            )

    final_front = fronts[0]
    assign_crowding_distance(final_front)
    history = pd.DataFrame(history_rows)
    return final_front, history, best_candidate


def train_surrogate_models(
    dataset: pd.DataFrame,
    feature_columns: list[str],
    random_state: int,
) -> tuple[VotingRegressor, RandomForestRegressor, RandomForestRegressor, RandomForestRegressor, dict[str, float], pd.DataFrame]:
    features = dataset[feature_columns].to_numpy(dtype=float)
    # Clip tiny negative P76 values (numerical noise near zero OCV point)
    current_target = np.clip(dataset[CURRENT_TARGET].to_numpy(dtype=float), 0.0, None)
    loss_target = dataset[LOSS_TARGET].to_numpy(dtype=float)
    fuel_out_target = dataset[FUEL_OUT_TARGET].to_numpy(dtype=float)
    efficiency_target = dataset[EFFICIENCY_TARGET].to_numpy(dtype=float)

    split = train_test_split(
        features,
        current_target,
        loss_target,
        fuel_out_target,
        efficiency_target,
        test_size=ML_TEST_SIZE,
        random_state=random_state,
    )
    x_train, x_test = split[0], split[1]
    current_train, current_test = split[2], split[3]
    loss_train, loss_test = split[4], split[5]
    fuel_out_train, fuel_out_test = split[6], split[7]
    efficiency_train, efficiency_test = split[8], split[9]

    rf_params: dict = dict(n_estimators=ML_RF_N_ESTIMATORS, random_state=random_state, n_jobs=-1, min_samples_leaf=ML_RF_MIN_SAMPLES_LEAF)

    # P76: ensemble of RF + GBR reduces MAE on extreme points while keeping R2 high
    current_model = VotingRegressor(
        estimators=[
            ("rf", RandomForestRegressor(n_estimators=ML_P76_RF_N_ESTIMATORS, random_state=random_state, n_jobs=-1, min_samples_leaf=ML_P76_RF_MIN_SAMPLES_LEAF)),
            ("gbr", GradientBoostingRegressor(n_estimators=ML_P76_GBR_N_ESTIMATORS, max_depth=ML_P76_GBR_MAX_DEPTH, learning_rate=ML_P76_GBR_LEARNING_RATE, random_state=random_state, subsample=ML_P76_GBR_SUBSAMPLE)),
        ],
        weights=ML_P76_VOTING_WEIGHTS,
    )
    loss_model = RandomForestRegressor(**rf_params)
    fuel_out_model = RandomForestRegressor(**rf_params)
    efficiency_model = RandomForestRegressor(**rf_params)

    current_model.fit(x_train, current_train)
    loss_model.fit(x_train, loss_train)
    fuel_out_model.fit(x_train, fuel_out_train)
    efficiency_model.fit(x_train, efficiency_train)

    current_pred = current_model.predict(x_test)
    loss_pred = loss_model.predict(x_test)
    fuel_out_pred = fuel_out_model.predict(x_test)
    efficiency_pred = efficiency_model.predict(x_test)
    metrics = {
        "current_r2": r2_score(current_test, current_pred),
        "current_mae": mean_absolute_error(current_test, current_pred),
        "loss_r2": r2_score(loss_test, loss_pred),
        "loss_mae": mean_absolute_error(loss_test, loss_pred),
        "fuel_out_r2": r2_score(fuel_out_test, fuel_out_pred),
        "fuel_out_mae": mean_absolute_error(fuel_out_test, fuel_out_pred),
        "efficiency_r2": r2_score(efficiency_test, efficiency_pred),
        "efficiency_mae": mean_absolute_error(efficiency_test, efficiency_pred),
    }
    accuracy_frame = pd.DataFrame(
        {
            "actual_P76": current_test,
            "predicted_P76": current_pred,
            "actual_P62": loss_test,
            "predicted_P62": loss_pred,
            "actual_P79": fuel_out_test,
            "predicted_P79": fuel_out_pred,
            "actual_P89": efficiency_test,
            "predicted_P89": efficiency_pred,
        }
    )

    current_model.fit(features, current_target)
    loss_model.fit(features, loss_target)
    fuel_out_model.fit(features, fuel_out_target)
    efficiency_model.fit(features, efficiency_target)
    return current_model, loss_model, fuel_out_model, efficiency_model, metrics, accuracy_frame


def dataset_pareto_front(dataset: pd.DataFrame) -> pd.DataFrame:
    candidates = dataset.reset_index(drop=True)
    pareto_mask = np.ones(len(candidates), dtype=bool)
    current_values = candidates[CURRENT_TARGET].to_numpy(dtype=float)
    loss_values = candidates[LOSS_TARGET].to_numpy(dtype=float)
    fuel_out_values = candidates[FUEL_OUT_TARGET].to_numpy(dtype=float)
    efficiency_values = candidates[EFFICIENCY_TARGET].to_numpy(dtype=float)

    for index in range(len(candidates)):
        # no_worse: P76 >=, P62 <=, P79 <=, P89 >=
        no_worse = (
            (current_values >= current_values[index])
            & (loss_values <= loss_values[index])
            & (fuel_out_values <= fuel_out_values[index])
            & (efficiency_values >= efficiency_values[index])
        )
        strictly_better = (
            (current_values > current_values[index])
            | (loss_values < loss_values[index])
            | (fuel_out_values < fuel_out_values[index])
            | (efficiency_values > efficiency_values[index])
        )
        dominating = no_worse & strictly_better
        dominating[index] = False
        if dominating.any():
            pareto_mask[index] = False

    front = candidates.loc[pareto_mask].copy()
    return front.sort_values([LOSS_TARGET, CURRENT_TARGET], ascending=[True, False]).reset_index(drop=True)


def normalized_distance_to_ideal(
    current_values: np.ndarray,
    loss_values: np.ndarray,
    fuel_out_values: np.ndarray | None = None,
    efficiency_values: np.ndarray | None = None,
) -> np.ndarray:
    def _safe_span(arr: np.ndarray) -> float:
        s = arr.max() - arr.min()
        return s if s else 1.0

    w = OBJECTIVE_WEIGHTS

    # P76: maximize -> score = (val - min) / span
    current_score = (current_values - current_values.min()) / _safe_span(current_values)
    # P62: minimize -> score = (max - val) / span
    loss_score = (loss_values.max() - loss_values) / _safe_span(loss_values)

    dist_sq = w[CURRENT_TARGET] * (1.0 - current_score) ** 2 + w[LOSS_TARGET] * (1.0 - loss_score) ** 2

    if fuel_out_values is not None:
        # P79: minimize -> score = (max - val) / span
        fuel_out_score = (fuel_out_values.max() - fuel_out_values) / _safe_span(fuel_out_values)
        dist_sq += w[FUEL_OUT_TARGET] * (1.0 - fuel_out_score) ** 2

    if efficiency_values is not None:
        # P89: maximize -> score = (val - min) / span
        efficiency_score = (efficiency_values - efficiency_values.min()) / _safe_span(efficiency_values)
        dist_sq += w[EFFICIENCY_TARGET] * (1.0 - efficiency_score) ** 2

    return np.sqrt(dist_sq)


def extract_compromise_candidate(front: list[Individual]) -> tuple[Individual, dict[str, float]]:
    current_values = np.array([ind.predicted_current for ind in front], dtype=float)
    loss_values = np.array([ind.predicted_losses for ind in front], dtype=float)
    fuel_out_values = np.array([ind.predicted_fuel_out for ind in front], dtype=float)
    efficiency_values = np.array([ind.predicted_efficiency for ind in front], dtype=float)
    distances = normalized_distance_to_ideal(current_values, loss_values, fuel_out_values, efficiency_values)
    best_index = int(np.argmin(distances))

    summary = {
        "front_size": float(len(front)),
        "best_distance_to_ideal": float(distances[best_index]),
        "best_predicted_P76": float(current_values[best_index]),
        "best_predicted_P62": float(loss_values[best_index]),
        "best_predicted_P79": float(fuel_out_values[best_index]),
        "best_predicted_P89": float(efficiency_values[best_index]),
        "front_max_P76": float(current_values.max()),
        "front_min_P62": float(loss_values.min()),
        "front_min_P79": float(fuel_out_values.min()),
        "front_max_P89": float(efficiency_values.max()),
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
    enriched["nearest_dataset_P79"] = nearest_rows[FUEL_OUT_TARGET]
    enriched["nearest_dataset_P89"] = nearest_rows[EFFICIENCY_TARGET]
    enriched["feature_distance_to_dataset"] = nearest_distances
    return enriched


def individuals_to_frame(front: list[Individual], dataset: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    data = []
    for individual in front:
        row = {column: value for column, value in zip(feature_columns, individual.genes)}
        row["predicted_P76"] = individual.predicted_current
        row["predicted_P62"] = individual.predicted_losses
        row["predicted_P79"] = individual.predicted_fuel_out
        row["predicted_P89"] = individual.predicted_efficiency
        row["rank"] = individual.rank
        row["crowding"] = individual.crowding
        data.append(row)

    pred_cols = ["predicted_P76", "predicted_P62", "predicted_P79", "predicted_P89"]
    frame = pd.DataFrame(data).drop_duplicates(subset=feature_columns + pred_cols)
    frame["distance_to_ideal"] = normalized_distance_to_ideal(
        frame["predicted_P76"].to_numpy(dtype=float),
        frame["predicted_P62"].to_numpy(dtype=float),
        frame["predicted_P79"].to_numpy(dtype=float),
        frame["predicted_P89"].to_numpy(dtype=float),
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
                "predicted_P79": individual.predicted_fuel_out,
                "predicted_P89": individual.predicted_efficiency,
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
    best_solution["surrogate_P79_R2"] = metrics["fuel_out_r2"]
    best_solution["surrogate_P79_MAE"] = metrics["fuel_out_mae"]
    best_solution["surrogate_P89_R2"] = metrics["efficiency_r2"]
    best_solution["surrogate_P89_MAE"] = metrics["efficiency_mae"]
    return best_solution


def plot_convergence(history: pd.DataFrame, output_path: Path) -> None:
    figure, axes = plt.subplots(5, 1, figsize=(10, 18), sharex=True)

    axes[0].plot(history["generation"], history["best_so_far_distance_to_ideal"], marker="o", color="#111827", linewidth=2)
    axes[0].set_title("GA convergence (4-objective)")
    axes[0].set_ylabel("Distance to ideal")
    axes[0].grid(alpha=0.3)

    axes[1].plot(history["generation"], history["front_max_P76"], marker="o", color="#2563eb", label="Front max P76")
    axes[1].plot(history["generation"], history["best_so_far_P76"], marker="s", color="#0f766e", label="Best-so-far P76")
    axes[1].set_ylabel("P76 current-density [A m^-2]")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(history["generation"], history["front_min_P62"], marker="o", color="#dc2626", label="Front min P62")
    axes[2].plot(history["generation"], history["best_so_far_P62"], marker="s", color="#7c2d12", label="Best-so-far P62")
    axes[2].set_ylabel("P62 total-losses [V]")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    axes[3].plot(history["generation"], history["front_min_P79"], marker="o", color="#7c3aed", label="Front min P79")
    axes[3].plot(history["generation"], history["best_so_far_P79"], marker="s", color="#4c1d95", label="Best-so-far P79")
    axes[3].set_ylabel("P79 fuel-out-h2")
    axes[3].legend()
    axes[3].grid(alpha=0.3)

    axes[4].plot(history["generation"], history["front_max_P89"], marker="o", color="#059669", label="Front max P89")
    axes[4].plot(history["generation"], history["best_so_far_P89"], marker="s", color="#064e3b", label="Best-so-far P89")
    axes[4].set_xlabel("Generation")
    axes[4].set_ylabel("P89 power-efficiency")
    axes[4].legend()
    axes[4].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_model_accuracy(accuracy_frame: pd.DataFrame, metrics: dict[str, float], output_path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(14, 12))
    plot_specs = [
        ("actual_P76", "predicted_P76", "P76 current-density", metrics["current_r2"], metrics["current_mae"], "#2563eb"),
        ("actual_P62", "predicted_P62", "P62 total-losses", metrics["loss_r2"], metrics["loss_mae"], "#dc2626"),
        ("actual_P79", "predicted_P79", "P79 fuel-out-h2", metrics["fuel_out_r2"], metrics["fuel_out_mae"], "#7c3aed"),
        ("actual_P89", "predicted_P89", "P89 power-efficiency", metrics["efficiency_r2"], metrics["efficiency_mae"], "#059669"),
    ]

    for axis, (actual_column, predicted_column, title, r2_value, mae_value, color) in zip(axes.flat, plot_specs):
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
    # 2x2 pairwise projections of the 4-objective Pareto front
    pair_specs = [
        (LOSS_TARGET, CURRENT_TARGET, "predicted_P62", "predicted_P76",
         "P62 total-losses [V]", "P76 current-density [A m^-2]"),
        (FUEL_OUT_TARGET, CURRENT_TARGET, "predicted_P79", "predicted_P76",
         "P79 fuel-out-h2", "P76 current-density [A m^-2]"),
        (LOSS_TARGET, EFFICIENCY_TARGET, "predicted_P62", "predicted_P89",
         "P62 total-losses [V]", "P89 power-efficiency"),
        (FUEL_OUT_TARGET, EFFICIENCY_TARGET, "predicted_P79", "predicted_P89",
         "P79 fuel-out-h2", "P89 power-efficiency"),
    ]

    figure, axes = plt.subplots(2, 2, figsize=(16, 14))
    for axis, (ds_x, ds_y, ga_x, ga_y, xlabel, ylabel) in zip(axes.flat, pair_specs):
        axis.scatter(dataset[ds_x], dataset[ds_y], alpha=0.2, s=25, color="#9ca3af", label="Dataset")
        axis.scatter(dataset_front[ds_x], dataset_front[ds_y], s=60, color="#2563eb", label="Dataset Pareto")
        axis.scatter(ga_front[ga_x], ga_front[ga_y], s=60, color="#dc2626", label="GA Pareto")
        if not ga_front.empty:
            best = ga_front.iloc[0]
            axis.scatter(best[ga_x], best[ga_y], s=140, color="#111827", zorder=5, label="GA best")
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)

    figure.suptitle("SOFC 4-objective Pareto front: pairwise projections", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


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
        default=GA_POPULATION_SIZE,
        help="GA population size. Increase it for a deeper search.",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=GA_GENERATIONS,
        help="Number of GA generations. Increase it for a deeper search.",
    )
    parser.add_argument("--crossover-rate", type=float, default=GA_CROSSOVER_RATE, help="Crossover rate.")
    parser.add_argument("--mutation-rate", type=float, default=GA_MUTATION_RATE, help="Per-gene mutation rate.")
    parser.add_argument(
        "--mutation-scale",
        type=float,
        default=GA_MUTATION_SCALE,
        help="Mutation sigma as a fraction of each feature range.",
    )
    parser.add_argument("--seed", type=int, default=GA_SEED, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = resolve_path(args.dataset, ROOT_DIR)
    output_path = resolve_path(args.output, ROOT_DIR)
    history_output_path = resolve_path(args.history_output, ROOT_DIR)
    convergence_figure_path = resolve_path(args.convergence_figure, ROOT_DIR)
    accuracy_figure_path = resolve_path(args.accuracy_figure, ROOT_DIR)

    dataset = pd.read_csv(dataset_path)
    required_columns = FEATURE_COLUMNS + ALL_TARGETS + ["source_file", "Name"]
    missing_columns = [column for column in required_columns if column not in dataset.columns]
    if missing_columns:
        raise KeyError(f"The dataset is missing required columns: {', '.join(missing_columns)}")

    modeling_dataset = dataset.dropna(subset=FEATURE_COLUMNS + ALL_TARGETS).copy()
    dropped_rows = len(dataset) - len(modeling_dataset)
    if modeling_dataset.empty:
        raise ValueError("No complete rows are available for GA training and optimization.")
    if dropped_rows:
        print(f"Rows excluded from modeling because of missing values: {dropped_rows}")

    current_model, loss_model, fuel_out_model, efficiency_model, metrics, accuracy_frame = train_surrogate_models(
        dataset=modeling_dataset,
        feature_columns=FEATURE_COLUMNS,
        random_state=args.seed,
    )

    print("Surrogate model validation metrics:")
    print(f"  P76 R2  = {metrics['current_r2']:.4f},  MAE = {metrics['current_mae']:.4f}")
    print(f"  P62 R2  = {metrics['loss_r2']:.4f},  MAE = {metrics['loss_mae']:.4f}")
    print(f"  P79 R2  = {metrics['fuel_out_r2']:.4f},  MAE = {metrics['fuel_out_mae']:.4f}")
    print(f"  P89 R2  = {metrics['efficiency_r2']:.4f},  MAE = {metrics['efficiency_mae']:.4f}")
    print(f"Decision variables used by the GA: {', '.join(FEATURE_COLUMNS)}")

    feature_space = modeling_dataset[FEATURE_COLUMNS].to_numpy(dtype=float)
    bounds = np.column_stack((feature_space.min(axis=0), feature_space.max(axis=0)))
    rng = np.random.default_rng(args.seed)

    front, convergence_history, best_candidate = evolve_population(
        current_model=current_model,
        loss_model=loss_model,
        fuel_out_model=fuel_out_model,
        efficiency_model=efficiency_model,
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