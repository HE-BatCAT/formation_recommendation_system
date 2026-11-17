"""Surrogate models that predict ageing metrics from the formation trajectory.

    features.py         formation solution -> 2 x 256 voltage/current samples
    dataset.py          builds surrogate/dataset.csv with PyBaMM (create_dataset.py)
    estimators/         XGBoost, GRU and custom PatchAttention models
    train_models.py     cross-validates and trains models
    predictor.py        loads a trained model for the MCO
    validate_protocol.py compares a trained model with PyBaMM on one protocol
"""

from surrogate.features import TrajectoryFeaturizer
from surrogate.predictor import SurrogatePredictor

__all__ = ["SurrogatePredictor", "TrajectoryFeaturizer"]
