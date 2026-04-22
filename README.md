# cvflow

A Python library for working with CV-flow (continuous variable flow) in measurement-based quantum computing (MBQC). It provides tools to define open graphs, check and find CV-flows, generate measurement patterns, and execute quantum computations.

## What is CV-flow?

In continuous variable MBQC, a quantum computation is described by an open graph $(G, I, O)$ where $G$ is a weighted graph, $I$ the input nodes, and $O$ the output nodes. A CV-flow is a function that assigns to each non-output node a correction vector, along with a layering of the graph, such that the measurement order is causally consistent.

`cvflow` implements algorithms to:
- Determine whether a CV-flow exists for a given open graph
- Compute the flow and measurement layers when it exists
- Generate executable measurement patterns from flows
- Execute patterns using different quantum backends

## Features

- **Graph Analysis**: Define and manipulate open graphs with input/output nodes
- **Flow Finding**: Automatic CV-flow detection with L1/L2 optimization methods
- **Pattern Generation**: Convert flows to executable measurement sequences
- **Multiple Backends**: Abstract backend interface with MrMustard implementation
- **Visualization**: Graph layout and visualization utilities

## Installation

Requires Python 3.11. Install with [uv](https://github.com/astral-sh/uv):

```bash
uv sync
```

For development (includes Sphinx, pytest):

```bash
uv sync --group dev
```

## Quick Start

### Finding a CV-flow

```python
import networkx as nx
from cvflow.graph import OpenGraph
from cvflow.flow import find_cvflow

# Create a graph
g = nx.Graph()
g.add_nodes_from([0, 1, 2, 3])
g.add_edge(0, 2, weight=1.0)
g.add_edge(1, 3, weight=1.0)
g.add_edge(2, 3, weight=1.0)

# Define the open graph
graph = OpenGraph(g, input_nodes=[0, 1], output_nodes=[2, 3])

# Find the CV-flow
exists, g_map, layers = find_cvflow(graph, method="l2")

if exists:
    print("CV-flow found!")
    print("Corrections:", g_map)
    print("Layers:", layers)
```

### Generating a Pattern

```python
from cvflow.pattern import flow_to_pattern

# Convert flow to an executable pattern
pattern = flow_to_pattern(graph, g_map, layers)
print(pattern)
```

### Checking a Specific Measurement Sequence

```python
from cvflow.flow import check_flow

measurements = [3, 2, 1]  # Measurement order
has_flow, g_map, layers = check_flow(graph, measurements, method="l2")

if has_flow:
    print("This measurement sequence satisfies the flow property")
```

## Project Structure

```
src/cvflow/
├── graph.py           # OpenGraph class and graph utilities
├── flow.py            # CV-flow finding and checking algorithms
├── pattern.py         # Pattern class and flow-to-pattern conversion
├── command.py         # Command types: N, M, E, X, Z

tests/                 # Test suite
```

## Testing

Run tests with pytest:

```bash
uv run pytest
```

## Methods

The flow-finding algorithms support two optimization methods:

- **L2** (default): Least squares minimization, spreads corrections across nodes
- **L1**: Linear programming for sparse corrections

```python
# Use L1 for sparser corrections
exists, g_map, layers = find_cvflow(graph, method="l1")
```

## Dependencies

- [NetworkX](https://networkx.org/) — graph structure and algorithms
- [NumPy](https://numpy.org/) — linear algebra operations
- [SciPy](https://scipy.org/) — optimization (L1 minimization)
- [Matplotlib](https://matplotlib.org/) — graph visualization
- [MrMustard](https://github.com/XanaduAI/MrMustard) — continuous variable quantum computing backend

## License

See LICENSE file for details.

## Author

Alexandre Menard (alexxandre.mnd+dev@gmail.com)
