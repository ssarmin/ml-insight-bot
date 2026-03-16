from model_service import TrainingResult, train_model


def test_train_model_returns_training_result() -> None:
    result = train_model()

    assert isinstance(result, TrainingResult)
    assert result.train_size > 0
    assert result.test_size > 0


def test_train_model_accuracy_above_threshold() -> None:
    result = train_model()

    assert result.accuracy >= 0.98


def test_train_model_logs_updated_accuracy(caplog) -> None:
    with caplog.at_level("INFO"):
        result = train_model()

    assert f"Updated accuracy: {result.accuracy:.4f}" in caplog.text
