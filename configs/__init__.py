"""configs package — ``ExperimentConfig`` and CLI helpers."""

from .base import (
    ExperimentConfig,
    MODEL_ALIASES,
    build_from_cli,
    routing_for,
    total_train_steps,
)

__all__ = [
    "ExperimentConfig",
    "MODEL_ALIASES",
    "build_from_cli",
    "routing_for",
    "total_train_steps",
]
