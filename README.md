# cvflow

A Python library for working with CV-flow (continuous variable flow) in measurement-based quantum computing (MBQC). It provides tools to define open graphs, check and find CV-flows, and visualize graph structures.

## What is CV-flow?

In continuous variable MBQC, a quantum computation is described by an open graph $(G, I, O)$ where $G$ is a weighted graph, $I$ the input nodes, and $O$ the output nodes. A CV-flow is a function that assigns to each non-output node a correction vector, along with a layering of the graph, such that the measurement order is causally consistent.

`cvflow` implements an algorithm to determine whether such a flow exists and to compute it when it does.

## Installation

Requires Python 3.13+. Install with [uv](https://github.com/astral-sh/uv):

```bash
uv sync
```

## Usage

```python
import networkx as nx
from cvflow.graph import OpenGraph
from cvflow.flow import find_cvflow

g = nx.Graph()
g.add_nodes_from([0, 1, 2, 3])
g.add_edge(0, 2, weight=1.0)
g.add_edge(1, 3, weight=1.0)
g.add_edge(2, 3, weight=1.0)

graph = OpenGraph(g, input_nodes=[0, 1], output_nodes=[2, 3])

exists, flow, layers = find_cvflow(graph)

if exists:
    print("CV-flow found")
    print("Layers:", layers)
```

## Structure

```
src/cvflow/
    graph.py    — OpenGraph class and layout utilities
    flow.py     — CV-flow existence check and finder
    pattern.py  — Pattern representation (sequence of commands)
    command.py  — Command types: N, M, E, X, Z
```

## Dependencies

- [NetworkX](https://networkx.org/) — graph structure
- [NumPy](https://numpy.org/) — linear algebra
- [Matplotlib](https://matplotlib.org/) — visualization
