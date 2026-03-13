from model_service import TrainingResult, train_model


def test_train_model_returns_training_result() -> None:
    result = train_model()

    assert isinstance(result, TrainingResult)
    assert result.train_size > 0
    assert result.test_size > 0


def test_train_model_accuracy_above_threshold() -> None:
    result = train_model()

    assert result.accuracy >= 0.90
