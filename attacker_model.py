"""
attacker_model.py

Implements the pipeline's "attacker model / internal service" step: given
anonymized trajectory data, estimate how likely an adversary is to
re-identify the individual behind it. This is the tool that stress-tests
an anonymization technique BEFORE it ships, not something aimed at real
people or real data.

Everything in this file operates on synthetically generated trajectories.
No real individuals, devices, or location datasets are used or represented.

Pipeline being modeled:
  1. Generate a synthetic population with realistic home/work commute
     patterns (`generate_population`).
  2. Anonymize each trajectory via spatial generalization — snapping GPS
     points to a coarser grid cell (`SpatialAnonymizer`). Larger cells =
     stronger anonymization, less precision.
  3. Attacker-side DAG reconstruction — because a coarse cell hides *where*
     within that cell someone actually was, the attacker builds a small
     directed graph of candidate fine-grained positions per timestep and
     finds the most physically plausible path through it (bounded by a
     max-speed constraint), analogous to the DAG-based trajectory
     reconstruction referenced in the pipeline design (`TrajectoryReconstructor`).
  4. Feature extraction from the reconstructed trajectory — inferred
     home/work centroids and a visited-cell histogram (`extract_features`).
  5. An `AttackerModel` (gradient-boosted classifier) trained on a
     labeled "background knowledge" population attempts to link each
     anonymized trajectory back to its true identity. Re-identification
     accuracy is the risk metric.

Swap `cell_size` (anonymization strength) and watch re-identification
accuracy fall — that's the actual privacy/utility tradeoff a real
anonymization pipeline has to navigate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

try:
    from xgboost import XGBClassifier  # optional — used if available
    _HAS_XGBOOST = True
except Exception:
    # Deliberately broad: xgboost can fail at import time for reasons other
    # than "not installed" — e.g. a version mismatch between an older
    # xgboost and a newer setuptools breaks its optional dask-detection
    # code with an AttributeError, not an ImportError. Since this backend
    # is optional and we fall back to scikit-learn either way, any failure
    # here should trigger the fallback rather than crash the whole module.
    _HAS_XGBOOST = False


# --------------------------------------------------------------------------
# Synthetic population + trajectory generation
# --------------------------------------------------------------------------

WORLD_SIZE = 100.0          # synthetic city is a WORLD_SIZE x WORLD_SIZE plane
NIGHT_HOURS = range(0, 7)   # treated as "at home"
DAY_HOURS = range(10, 17)   # treated as "at work"
MAX_PLAUSIBLE_SPEED = 15.0  # world-units per hour, used as a physical constraint


@dataclass
class SyntheticIndividual:
    identity: int
    home: tuple[float, float]
    work: tuple[float, float]


@dataclass
class Ping:
    hour: int          # hour-of-day, 0-23, repeated across days
    x: float
    y: float


def _commute_position(hour: int, home: tuple[float, float], work: tuple[float, float], rng: np.random.Generator):
    """Where a person plausibly is at a given hour, plus movement noise."""
    if hour in NIGHT_HOURS:
        base = home
    elif hour in DAY_HOURS:
        base = work
    elif hour in (7, 8, 9):
        frac = (hour - 6) / 4.0  # commuting toward work
        base = (home[0] + frac * (work[0] - home[0]), home[1] + frac * (work[1] - home[1]))
    elif hour in (17, 18, 19):
        frac = (hour - 16) / 4.0  # commuting back home
        base = (work[0] + frac * (home[0] - work[0]), work[1] + frac * (home[1] - work[1]))
    else:
        base = home  # evening at home
    noise = rng.normal(scale=1.5, size=2)
    return float(base[0] + noise[0]), float(base[1] + noise[1])


def generate_population(
    n_individuals: int = 60,
    n_days: int = 5,
    seed: int = 0,
) -> tuple[list[SyntheticIndividual], dict[int, list[list[Ping]]]]:
    """
    Generate a synthetic population with distinct home/work locations and
    hourly location pings following a realistic commute pattern.

    Returns trajectories as `identity -> list of daily trajectories`
    (one list of 24 hourly Pings per day), so each identity yields
    multiple independent samples — matching the standard re-identification
    attack setup where an adversary observes several days of released data.
    """
    rng = np.random.default_rng(seed)
    individuals = []
    trajectories: dict[int, list[list[Ping]]] = {}

    for identity in range(n_individuals):
        home = tuple(rng.uniform(5, WORLD_SIZE - 5, size=2))
        work = tuple(rng.uniform(5, WORLD_SIZE - 5, size=2))
        individuals.append(SyntheticIndividual(identity, home, work))

        daily_trajectories = []
        for _day in range(n_days):
            pings = []
            for hour in range(24):
                x, y = _commute_position(hour, home, work, rng)
                pings.append(Ping(hour=hour, x=x, y=y))
            daily_trajectories.append(pings)
        trajectories[identity] = daily_trajectories

    return individuals, trajectories


# --------------------------------------------------------------------------
# Anonymization
# --------------------------------------------------------------------------

class SpatialAnonymizer:
    """
    Snaps each ping to the centroid of a `cell_size` x `cell_size` grid
    cell. Larger cell_size = stronger anonymization, at the cost of
    utility (the released data says less about where someone actually was).
    """

    def __init__(self, cell_size: float):
        self.cell_size = cell_size

    def anonymize(self, pings: list[Ping]) -> list[Ping]:
        anonymized = []
        for p in pings:
            cell_x = math.floor(p.x / self.cell_size) * self.cell_size + self.cell_size / 2
            cell_y = math.floor(p.y / self.cell_size) * self.cell_size + self.cell_size / 2
            anonymized.append(Ping(hour=p.hour, x=cell_x, y=cell_y))
        return anonymized


# --------------------------------------------------------------------------
# Attacker-side DAG reconstruction
# --------------------------------------------------------------------------

class TrajectoryReconstructor:
    """
    Attempts to undo spatial generalization: for each anonymized ping, the
    true position could be anywhere in that grid cell. We sample a handful
    of candidate fine-grained positions per timestep and find the path
    through consecutive timesteps that best respects a max-speed
    constraint — i.e. a shortest-path search over a small DAG where nodes
    are (timestep, candidate position) and edges only exist between
    physically plausible transitions.
    """

    def __init__(self, cell_size: float, n_candidates: int = 4, max_speed: float = MAX_PLAUSIBLE_SPEED, seed: int = 0):
        self.cell_size = cell_size
        self.n_candidates = n_candidates
        self.max_speed = max_speed
        self._rng = np.random.default_rng(seed)

    def _candidates_for(self, ping: Ping) -> np.ndarray:
        """Sample candidate true positions within the anonymized cell."""
        half = self.cell_size / 2
        offsets = self._rng.uniform(-half, half, size=(self.n_candidates, 2))
        return np.array([ping.x, ping.y]) + offsets

    def reconstruct(self, anonymized_pings: list[Ping]) -> list[tuple[float, float]]:
        """
        Dynamic-programming (Viterbi-style) search over the DAG of candidate
        positions. Returns the single most plausible fine-grained path.
        """
        n_steps = len(anonymized_pings)
        candidates = [self._candidates_for(p) for p in anonymized_pings]

        # cost[t][i] = min cumulative implausibility to reach candidate i at step t
        cost = [np.zeros(self.n_candidates)]
        backpointer: list[np.ndarray] = [np.full(self.n_candidates, -1)]

        for t in range(1, n_steps):
            dt_hours = max(anonymized_pings[t].hour - anonymized_pings[t - 1].hour, 1) % 24 or 1
            prev_points = candidates[t - 1]
            curr_points = candidates[t]

            step_cost = np.full(self.n_candidates, np.inf)
            step_backptr = np.full(self.n_candidates, -1)

            for j in range(self.n_candidates):
                dists = np.linalg.norm(prev_points - curr_points[j], axis=1)
                implied_speed = dists / dt_hours
                transition_cost = np.where(
                    implied_speed <= self.max_speed, dists, dists + 1000.0  # heavy penalty, not a hard cutoff
                )
                total = cost[t - 1] + transition_cost
                best_prev = int(np.argmin(total))
                step_cost[j] = total[best_prev]
                step_backptr[j] = best_prev

            cost.append(step_cost)
            backpointer.append(step_backptr)

        # backtrack from the cheapest final node
        path_indices = [int(np.argmin(cost[-1]))]
        for t in range(n_steps - 1, 0, -1):
            path_indices.append(int(backpointer[t][path_indices[-1]]))
        path_indices.reverse()

        return [tuple(candidates[t][idx]) for t, idx in enumerate(path_indices)]


# --------------------------------------------------------------------------
# Feature extraction
# --------------------------------------------------------------------------

def extract_features(pings: list[Ping], reconstructed: Optional[list[tuple[float, float]]] = None) -> np.ndarray:
    """
    Turn a (reconstructed) trajectory into a fixed-length feature vector:
    inferred home centroid, inferred work centroid, and a coarse visited-
    cell histogram — the same signal a real linkage attack would use.
    """
    points = reconstructed if reconstructed is not None else [(p.x, p.y) for p in pings]
    points_arr = np.array(points)
    hours = np.array([p.hour for p in pings])

    night_mask = np.isin(hours, list(NIGHT_HOURS))
    day_mask = np.isin(hours, list(DAY_HOURS))

    home_est = points_arr[night_mask].mean(axis=0) if night_mask.any() else points_arr.mean(axis=0)
    work_est = points_arr[day_mask].mean(axis=0) if day_mask.any() else points_arr.mean(axis=0)

    # coarse 5x5 visited-cell histogram as extra signal
    grid_n = 5
    grid_size = WORLD_SIZE / grid_n
    hist = np.zeros(grid_n * grid_n)
    for x, y in points_arr:
        gx = min(int(x // grid_size), grid_n - 1)
        gy = min(int(y // grid_size), grid_n - 1)
        hist[gx * grid_n + gy] += 1
    hist = hist / hist.sum() if hist.sum() > 0 else hist

    return np.concatenate([home_est, work_est, hist])


# --------------------------------------------------------------------------
# Attacker model
# --------------------------------------------------------------------------

class AttackerModel:
    """
    Wraps a classifier trained to link anonymized/reconstructed trajectory
    features back to a true identity, using a labeled "background
    knowledge" population — the standard assumption in re-identification
    attack literature (the attacker has seen some prior data linking
    identities to movement patterns).

    Uses gradient-boosted trees (XGBoost if installed, otherwise
    scikit-learn's GradientBoostingClassifier) to mirror the modeling
    approach referenced in the pipeline design.
    """

    def __init__(self, use_xgboost: Optional[bool] = None, seed: int = 0):
        use_xgboost = _HAS_XGBOOST if use_xgboost is None else (use_xgboost and _HAS_XGBOOST)
        if use_xgboost:
            self._model = XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.1,
                eval_metric="mlogloss", random_state=seed,
            )
        else:
            self._model = GradientBoostingClassifier(
                n_estimators=200, max_depth=3, learning_rate=0.1, random_state=seed,
            )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AttackerModel":
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def reidentification_accuracy(self, X: np.ndarray, y_true: np.ndarray) -> float:
        return float(accuracy_score(y_true, self.predict(X)))


# --------------------------------------------------------------------------
# End-to-end risk evaluation
# --------------------------------------------------------------------------

def evaluate_reidentification_risk(
    cell_size: float,
    n_individuals: int = 60,
    n_days: int = 5,
    reconstruct: bool = True,
    seed: int = 0,
) -> float:
    """
    Full pipeline: generate a synthetic population, anonymize at the given
    strength, (optionally) run DAG-based reconstruction, extract features,
    train/test split, and report re-identification accuracy — the risk
    metric a real anonymization review would gate a launch decision on.
    """
    _individuals, trajectories = generate_population(n_individuals, n_days, seed=seed)
    anonymizer = SpatialAnonymizer(cell_size)
    reconstructor = TrajectoryReconstructor(cell_size, seed=seed) if reconstruct else None

    X, y = [], []
    for identity, daily_trajectories in trajectories.items():
        for pings in daily_trajectories:
            anonymized = anonymizer.anonymize(pings)
            reconstructed = reconstructor.reconstruct(anonymized) if reconstructor else None
            X.append(extract_features(anonymized, reconstructed))
            y.append(identity)

    X, y = np.array(X), np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=seed, stratify=y
    )

    model = AttackerModel(seed=seed).fit(X_train, y_train)
    return model.reidentification_accuracy(X_test, y_test)


def _demo() -> None:
    print(f"gradient boosting backend: {'XGBoost' if _HAS_XGBOOST else 'scikit-learn (XGBoost not installed)'}\n")
    print(f"{'cell_size':>10} | {'re-id accuracy':>15}")
    print("-" * 30)
    for cell_size in (2, 10, 25, 50, 80):
        risk = evaluate_reidentification_risk(cell_size=cell_size, n_individuals=60, seed=0)
        print(f"{cell_size:>10} | {risk:>14.1%}")

    print(
        "\nAs expected: larger grid cells (stronger spatial generalization) "
        "drive re-identification accuracy down — this is the privacy/utility "
        "tradeoff curve a real anonymization config decision has to sit on."
    )


if __name__ == "__main__":
    _demo()
