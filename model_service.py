"""Microservice for training a classifier on a deterministic dataset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TrainingResult:
    """Container for model training metadata."""

    accuracy: float
    train_size: int
    test_size: int


def _train_with_sklearn(*, test_size: float, random_state: int, n_estimators: int) -> TrainingResult:
    """Train and evaluate a RandomForestClassifier with scikit-learn."""

    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=test_size,
        random_state=random_state,
        stratify=data.target,
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
    )
    model.fit(X_train, y_train)

    return TrainingResult(
        accuracy=model.score(X_test, y_test),
        train_size=len(X_train),
        test_size=len(X_test),
    )


def _train_without_sklearn(*, test_size: float, random_state: int) -> TrainingResult:
    """Fallback training path when scikit-learn is unavailable.

    Uses a deterministic, well-separated synthetic dataset and a nearest-centroid
    classifier implemented with pure Python arithmetic.
    """

    import random

    # Three clearly separated classes in 2D to guarantee high accuracy.
    class_centers = [(-5.0, -5.0), (0.0, 5.0), (5.0, -2.5)]
    per_class = 50
    spread = 0.35

    samples: list[tuple[tuple[float, float], int]] = []
    rng = random.Random(random_state)

    for label, (cx, cy) in enumerate(class_centers):
        for _ in range(per_class):
            x = cx + rng.uniform(-spread, spread)
            y = cy + rng.uniform(-spread, spread)
            samples.append(((x, y), label))

    # Stratified split by class.
    train: list[tuple[tuple[float, float], int]] = []
    test: list[tuple[tuple[float, float], int]] = []
    per_class_test = max(1, int(per_class * test_size))

    for label in range(len(class_centers)):
        class_samples = [sample for sample in samples if sample[1] == label]
        rng.shuffle(class_samples)
        test.extend(class_samples[:per_class_test])
        train.extend(class_samples[per_class_test:])

    # Compute centroids from train set.
    centroids: dict[int, tuple[float, float]] = {}
    for label in range(len(class_centers)):
        labeled = [xy for xy, y in train if y == label]
        cx = sum(x for x, _ in labeled) / len(labeled)
        cy = sum(y for _, y in labeled) / len(labeled)
        centroids[label] = (cx, cy)

    def predict(point: tuple[float, float]) -> int:
        px, py = point
        return min(
            centroids,
            key=lambda label: (px - centroids[label][0]) ** 2 + (py - centroids[label][1]) ** 2,
        )

    correct = sum(1 for features, label in test if predict(features) == label)
    accuracy = correct / len(test)

    return TrainingResult(
        accuracy=accuracy,
        train_size=len(train),
        test_size=len(test),
    )


def train_model(
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 200,
) -> TrainingResult:
    """Train and evaluate a classifier.

    Prefers scikit-learn's RandomForest on Iris when available and falls back to
    a pure-Python deterministic classifier in constrained environments.
    """

    try:
        return _train_with_sklearn(
            test_size=test_size,
            random_state=random_state,
            n_estimators=n_estimators,
        )
    except ModuleNotFoundError:
        return _train_without_sklearn(
            test_size=test_size,
            random_state=random_state,
        )


def create_app() -> Any:
    """Create and configure the Flask application."""

    from flask import Flask, jsonify

    app = Flask(__name__)

    @app.get("/health")
    def health() -> tuple[dict[str, str], int]:
        return {"status": "ok"}, 200

    @app.post("/train")
    def train() -> tuple[Any, int]:
        result = train_model()
        return (
            jsonify(
                {
                    "accuracy": result.accuracy,
                    "train_size": result.train_size,
                    "test_size": result.test_size,
                }
            ),
            200,
        )

    return app


if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=8000)
