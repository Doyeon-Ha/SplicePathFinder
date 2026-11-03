"""Evaluation package for IssaSplice model.

Provides :func:`evaluate.evaluate.evaluate` as a CLI
entrypoint and :class:`evaluate.evaluator.Evaluator`
as a standalone evaluator free of SplicingTrainer.
"""

from evaluate.evaluate import evaluate
from evaluate.evaluator import Evaluator

__all__ = ["evaluate", "Evaluator"]
