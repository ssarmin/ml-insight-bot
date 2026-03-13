"""Microservice for training a RandomForest model on the Iris dataset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class TrainingResult:
    """Container for model training metadata."""

    accuracy: float
    train_size: int
    test_size: int


def train_model(
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 200,
) -> TrainingResult:
    """Train and evaluate a RandomForestClassifier on the Iris dataset."""

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

    accuracy = model.score(X_test, y_test)

    return TrainingResult(
        accuracy=accuracy,
        train_size=len(X_train),
        test_size=len(X_test),
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
