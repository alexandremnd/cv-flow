"""cvflow - Continuous Variable Flow for Measurement-Based Quantum Computing.

A Python library for working with CV-flow in MBQC, providing tools to define
open graphs, check and find CV-flows, generate measurement patterns, and execute
quantum computations with publication-quality visualization.
"""

from cvflow.graph import OpenGraph
from cvflow.flow import find_cvflow, check_flow
from cvflow.pattern import Pattern, flow_to_pattern
from cvflow.command import N, M, E, X, Z

__version__ = "0.1.0"

__all__ = [
    # Core graph functionality
    "OpenGraph",
    # Flow algorithms
    "find_cvflow",
    "check_flow",
    # Pattern generation
    "Pattern",
    "flow_to_pattern",
    # Commands
    "N",
    "M",
    "E",
    "X",
    "Z",
]


def main() -> None:
    print(f"Version: {__version__}")
