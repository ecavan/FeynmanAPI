"""Amplitude backends and shared amplitude result types."""

from feynman_engine.amplitudes.types import AmplitudeResult
from feynman_engine.amplitudes.symbolic import (
    get_symbolic_amplitude,
    get_loop_integral_latex,
    get_tree_integral_latex,
)

__all__ = ["AmplitudeResult", "get_symbolic_amplitude", "get_loop_integral_latex", "get_tree_integral_latex"]
