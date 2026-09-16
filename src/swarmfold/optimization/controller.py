from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np


@dataclass(frozen=True)
class HPOConfig:
    study_id: str
    direction: Literal["maximize", "minimize"]
    seed: int = 42
    startup_trials: int = 5
    elite_fraction: float = 0.25
    min_trials_before_pruning: int = 5


@dataclass(frozen=True)
class PruneDecision:
    should_prune: bool
    fold_index: int
    score: float
    reference_median: float | None
    reason: str


_SEARCH_SPACES: dict[str, dict[str, tuple[Any, ...]]] = {
    "hist_gradient_boosting": {
        "max_iter": ("int", 75, 500),
        "max_depth": ("int", 2, 14),
        "learning_rate": ("float", 0.01, 0.2, "log"),
        "l2_regularization": ("float", 1e-4, 20.0, "log"),
        "min_samples_leaf": ("int", 10, 100),
    },
    "lightgbm": {
        "n_estimators": ("int", 150, 1500),
        "max_depth": ("int", 3, 14),
        "learning_rate": ("float", 0.01, 0.2, "log"),
        "num_leaves": ("int", 15, 255),
        "subsample": ("float", 0.6, 1.0, "linear"),
        "colsample_bytree": ("float", 0.6, 1.0, "linear"),
        "reg_lambda": ("float", 1e-4, 20.0, "log"),
    },
    "xgboost": {
        "n_estimators": ("int", 150, 1500),
        "max_depth": ("int", 3, 12),
        "learning_rate": ("float", 0.01, 0.2, "log"),
        "subsample": ("float", 0.6, 1.0, "linear"),
        "colsample_bytree": ("float", 0.6, 1.0, "linear"),
        "min_child_weight": ("float", 0.5, 20.0, "log"),
        "reg_lambda": ("float", 1e-4, 20.0, "log"),
    },
    "catboost": {
        "n_estimators": ("int", 150, 1500),
        "max_depth": ("int", 4, 12),
        "learning_rate": ("float", 0.01, 0.2, "log"),
        "l2_leaf_reg": ("float", 0.1, 30.0, "log"),
        "random_strength": ("float", 1e-4, 10.0, "log"),
    },
}


class HPOController:
    """Persistent adaptive sampler with same-fold median pruning."""

    def __init__(self, storage_path: Path, config: HPOConfig) -> None:
        self.storage_path = storage_path
        self.config = config
        storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @property
    def higher_is_better(self) -> bool:
        return self.config.direction == "maximize"

    def suggest(self, model_family: str, trial_number: int) -> dict[str, Any]:
        if model_family not in _SEARCH_SPACES:
            raise ValueError(f"No adaptive search space for '{model_family}'")
        existing = self._trial(trial_number)
        if existing is not None:
            if existing["model_family"] != model_family:
                raise ValueError("Existing HPO trial uses a different model family")
            return dict(existing["parameters"])

        rng = np.random.default_rng(self.config.seed + trial_number * 104729)
        completed = self._completed_trials(model_family)
        elite = self._elite_trials(completed)
        parameters = {
            name: self._sample_distribution(specification, rng, elite, name)
            for name, specification in _SEARCH_SPACES[model_family].items()
        }
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO hpo_trials (
                    study_id, trial_number, model_family, parameters_json, status,
                    fold_scores_json, objective
                ) VALUES (?, ?, ?, ?, 'SUGGESTED', '[]', NULL)
                """,
                (
                    self.config.study_id,
                    trial_number,
                    model_family,
                    json.dumps(parameters, sort_keys=True),
                ),
            )
        return parameters

    def report_fold(
        self,
        trial_number: int,
        fold_index: int,
        score: float,
    ) -> PruneDecision:
        trial = self._required_trial(trial_number)
        fold_scores = list(trial["fold_scores"])
        if len(fold_scores) > fold_index:
            if not math.isclose(float(fold_scores[fold_index]), float(score)):
                raise ValueError("Repeated HPO fold report changed its score")
            references = self._same_step_scores(fold_index, exclude_trial=trial_number)
            reference = (
                float(np.median(references))
                if len(references) >= self.config.min_trials_before_pruning
                else None
            )
            prune = reference is not None and (
                score < reference if self.higher_is_better else score > reference
            )
            return PruneDecision(
                should_prune=prune,
                fold_index=fold_index,
                score=float(score),
                reference_median=reference,
                reason=(
                    "Repeated fold report reproduced the pruning decision"
                    if reference is not None
                    else "Repeated fold report has insufficient pruning evidence"
                ),
            )
        if len(fold_scores) != fold_index:
            raise ValueError("HPO folds must be reported once and in order")
        fold_scores.append(float(score))
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE hpo_trials SET fold_scores_json = ?, status = 'RUNNING'
                WHERE study_id = ? AND trial_number = ?
                """,
                (json.dumps(fold_scores), self.config.study_id, trial_number),
            )

        references = self._same_step_scores(fold_index, exclude_trial=trial_number)
        if len(references) < self.config.min_trials_before_pruning:
            return PruneDecision(
                should_prune=False,
                fold_index=fold_index,
                score=float(score),
                reference_median=None,
                reason="Insufficient same-fold evidence for pruning",
            )
        reference = float(np.median(references))
        prune = score < reference if self.higher_is_better else score > reference
        return PruneDecision(
            should_prune=prune,
            fold_index=fold_index,
            score=float(score),
            reference_median=reference,
            reason=(
                "Trial is worse than the completed same-fold median"
                if prune
                else "Trial meets or exceeds the completed same-fold median"
            ),
        )

    def complete(self, trial_number: int, objective: float) -> None:
        self._set_terminal(trial_number, "COMPLETE", objective)

    def prune(self, trial_number: int, objective: float) -> None:
        self._set_terminal(trial_number, "PRUNED", objective)

    def fail(self, trial_number: int) -> None:
        self._set_terminal(trial_number, "FAILED", None)

    def trials(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT trial_number, model_family, parameters_json, status,
                       fold_scores_json, objective
                FROM hpo_trials WHERE study_id = ? ORDER BY trial_number
                """,
                (self.config.study_id,),
            ).fetchall()
        return [self._row_to_trial(row) for row in rows]

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS hpo_studies (
                    study_id TEXT PRIMARY KEY,
                    direction TEXT NOT NULL,
                    seed INTEGER NOT NULL,
                    config_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS hpo_trials (
                    study_id TEXT NOT NULL,
                    trial_number INTEGER NOT NULL,
                    model_family TEXT NOT NULL,
                    parameters_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    fold_scores_json TEXT NOT NULL,
                    objective REAL,
                    PRIMARY KEY (study_id, trial_number)
                )
                """
            )
            row = connection.execute(
                "SELECT direction, seed FROM hpo_studies WHERE study_id = ?",
                (self.config.study_id,),
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO hpo_studies VALUES (?, ?, ?, ?)",
                    (
                        self.config.study_id,
                        self.config.direction,
                        self.config.seed,
                        json.dumps(self.config.__dict__, sort_keys=True),
                    ),
                )
            elif row != (self.config.direction, self.config.seed):
                raise ValueError("Stored HPO study configuration is incompatible")

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.storage_path)

    def _trial(self, trial_number: int) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT trial_number, model_family, parameters_json, status,
                       fold_scores_json, objective
                FROM hpo_trials WHERE study_id = ? AND trial_number = ?
                """,
                (self.config.study_id, trial_number),
            ).fetchone()
        return self._row_to_trial(row) if row is not None else None

    def _required_trial(self, trial_number: int) -> dict[str, Any]:
        trial = self._trial(trial_number)
        if trial is None:
            raise ValueError(f"Unknown HPO trial {trial_number}")
        return trial

    @staticmethod
    def _row_to_trial(row: tuple[Any, ...]) -> dict[str, Any]:
        return {
            "trial_number": int(row[0]),
            "model_family": str(row[1]),
            "parameters": json.loads(row[2]),
            "status": str(row[3]),
            "fold_scores": json.loads(row[4]),
            "objective": float(row[5]) if row[5] is not None else None,
        }

    def _completed_trials(self, model_family: str) -> list[dict[str, Any]]:
        return [
            trial
            for trial in self.trials()
            if trial["model_family"] == model_family and trial["status"] == "COMPLETE"
        ]

    def _elite_trials(self, completed: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if len(completed) < self.config.startup_trials:
            return []
        ordered = sorted(
            completed,
            key=lambda trial: float(trial["objective"]),
            reverse=self.higher_is_better,
        )
        count = max(1, math.ceil(len(ordered) * self.config.elite_fraction))
        return ordered[:count]

    @staticmethod
    def _sample_distribution(
        specification: tuple[Any, ...],
        rng: np.random.Generator,
        elite: list[dict[str, Any]],
        parameter_name: str,
    ) -> int | float:
        kind, low, high, *scale = specification
        mode = scale[0] if scale else "linear"
        elite_values = [trial["parameters"][parameter_name] for trial in elite]
        if elite_values:
            center = float(elite_values[int(rng.integers(0, len(elite_values)))])
            if mode == "log":
                spread = (math.log(high) - math.log(low)) / 6.0
                value = math.exp(rng.normal(math.log(center), spread))
            else:
                value = rng.normal(center, (high - low) / 6.0)
        elif mode == "log":
            value = math.exp(rng.uniform(math.log(low), math.log(high)))
        else:
            value = rng.uniform(low, high)
        value = min(max(value, low), high)
        return int(round(value)) if kind == "int" else float(value)

    def _same_step_scores(self, fold_index: int, exclude_trial: int) -> list[float]:
        scores = []
        for trial in self.trials():
            if trial["trial_number"] == exclude_trial:
                continue
            fold_scores = trial["fold_scores"]
            if len(fold_scores) > fold_index:
                scores.append(float(fold_scores[fold_index]))
        return scores

    def _set_terminal(
        self,
        trial_number: int,
        status: str,
        objective: float | None,
    ) -> None:
        self._required_trial(trial_number)
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE hpo_trials SET status = ?, objective = ?
                WHERE study_id = ? AND trial_number = ?
                """,
                (status, objective, self.config.study_id, trial_number),
            )
