from collections import defaultdict
from typing import assert_never

from cvflow.command import Command, CommandKind, N, E, M, X, Z, Node
from cvflow.graph import OpenGraph

import numpy as np

class Pattern:
    """A sequence of commands applied to a graph state.

    Attributes
    ----------
    _commands : list[Command]
        The list of commands in the pattern.
    """
    def __init__(self, commands: list[Command], input_nodes: list[Node]):
        self._commands = commands
        self._input_nodes = set(input_nodes)

        self.check_runnability()

    def check_runnability(self):
        """Check if the pattern is runnable.

        Verifies that the commands can be applied to a graph state without
        contradictions, which are defined as:

        - a preparation command (N) is not applied to a node that has
          already been measured.
        - a non preparation command (E, M, X, Z) is not applied to a node
          that has not been prepared or already measured.
        - correction domains (M, X, Z) do not refer to a node that has not
          been measured.

        Raises
        ------
        ValueError
            If the pattern is not valid according to the rules of command
            application.
        """
        initialised_nodes = self._input_nodes.copy()
        measured_nodes = set()

        for i, cmd in enumerate(self._commands):
            if cmd.kind == CommandKind.N:
                if cmd.node in initialised_nodes:
                    print(self)
                    raise ValueError(f"Node {cmd.node} is already initialised.")
                initialised_nodes.add(cmd.node)
            elif cmd.kind == CommandKind.E:
                if cmd.nodes[0] not in initialised_nodes or cmd.nodes[1] not in initialised_nodes:
                    print(self)
                    raise ValueError(f"Entanglement command {cmd} ({i+1}-th command) requires both nodes to be initialised.")
            elif cmd.kind == CommandKind.M:
                if cmd.node not in initialised_nodes:
                    print(self)
                    raise ValueError(f"Measurement command {cmd} ({i+1}-th command) requires node {cmd.node} to be initialised.")
                if non_measured_nodes := (cmd.x_domain.keys() | cmd.z_domain.keys()) - measured_nodes:
                    raise ValueError(f"Measurement command {cmd} ({i+1}-th command) requires node(s) {",".join(non_measured_nodes)} to be measured.")
                initialised_nodes.remove(cmd.node)
                measured_nodes.add(cmd.node)
            elif cmd.kind in (CommandKind.X, CommandKind.Z):
                if cmd.node not in initialised_nodes:
                    print(self)
                    raise ValueError(f"Correction command {cmd} ({i+1}-th command) requires node {cmd.node} to be initialised.")
                domain = cmd.x_domain.keys() if cmd.kind == CommandKind.X else cmd.z_domain.keys()
                if non_measured_nodes := domain - measured_nodes:
                    raise ValueError(f"Correction command {cmd} ({i+1}-th command) requires node(s) {",".join(non_measured_nodes)} to be measured.")
            else:
                assert_never(cmd.kind)

    def append(self, cmd: Command):
        """Append a command to the pattern (it will be the latest to be executed).

        Parameters
        ----------
        cmd : Command
            The command to be appended to the pattern.

        Raises
        ------
        ValueError
            If the resulting pattern is not valid after appending the command.
        """
        self._commands.append(cmd)
        # Note that `check_runnability` runs in O(n). Calling this
        # each time a command is appended results in O(n²) complexity
        # for `append`.
        self.check_runnability()

    def insert(self, index: int, cmd: Command):
        """Insert a command at a specific position in the pattern.

        Parameters
        ----------
        index : int
            The position at which to insert the command (0-based index).
        cmd : Command
            The command to be inserted.

        Raises
        ------
        ValueError
            If the resulting pattern is not valid after inserting the command.
        """
        self._commands.insert(index, cmd)
        self.check_runnability()

    def __str__(self):
        num_commands = len(self._commands)
        width = len(str(num_commands))
        return "\n".join(f"{i+1:0{width}d}) {cmd}" for i, cmd in enumerate(self._commands))


def flow_to_pattern(graph: OpenGraph, g: dict[int, dict[int, float]], layer: dict[int, list[int]], squeezing_params: dict[int, tuple[float, float]] = {}) -> Pattern:
    """Convert an OpenGraph with a flow to a Pattern.

    This function takes a flow and converts it into a Pattern by generating
    the appropriate sequence of commands based on the flow structure. The
    conversion follows this order: node preparation (N), entanglement (E),
    measurements (M) by layer, and corrections (X, Z) for each layer.

    Parameters
    ----------
    graph : OpenGraph
        The input graph containing the nodes and edges to be converted.
    g : dict[int, dict[int, float]]
        The flow correction mapping. For each measured node, maps to a
        dictionary of target nodes and their correction amplitudes.
        Structure: {measured_node: {target_node: amplitude}}.
    layer : dict[int, list[int]]
        The layering of nodes for measurement order. Maps layer numbers
        to lists of nodes that should be measured in that layer.
        Structure: {layer_num: [node1, node2, ...]}.
    squeezing_params : dict[int, tuple[float, float]], optional
        Optional squeezing parameters for each node. Maps node numbers to
        tuples of (squeezing_ratio, squeezing_angle). If not provided, defaults
        to a squeezing ratio of 2.0 and angle of π/2 for all nodes

    Returns
    ------
    Pattern
        The resulting pattern corresponding to the input graph's flow,
        containing the complete sequence of commands (N, E, M, X, Z)
        that implement the measurement-based quantum computation.
    """
    command_list: list[Command] = []

    # Prepare all nodes
    for node_to_measure in graph.non_input_nodes:
        squeezing_ratio, squeezing_angle = squeezing_params.get(node_to_measure, (2.0, np.pi/2))
        command_list.append(N(node_to_measure, squeezing_ratio=squeezing_ratio, squeezing_angle=squeezing_angle))  # type: ignore

    # Entangle according to the graph edges
    for node1, node2, weight in graph.edges:
        command_list.append(E((node1, node2), weight))

    measured_nodes = set()
    max_layer = max(layer.keys())
    for layer_idx in reversed(range(1, max_layer)):
        # Measure nodes in the current layer
        for node_to_measure in layer[layer_idx]:
            command_list.append(M(node_to_measure))
            measured_nodes.add(node_to_measure)

        # dict[node_receiving_correction, dict[node_to_correct, amplitude]]
        z_corrections: dict[Node, dict[Node, float]] = defaultdict(dict)
        x_corrections: dict[Node, dict[Node, float]] = defaultdict(dict)
        for node_to_correct in layer[layer_idx]:
            corrections = g.get(node_to_correct, {})

            for node_receiving, amplitude in corrections.items():
                x_corrections[node_receiving][node_to_correct] = amplitude

                # Z part applied on the neighbor of the node receiving the X correction
                for neighbour in graph.get_neighbours(node_receiving):
                    if neighbour != node_to_correct and neighbour not in measured_nodes:
                        z_corrections[neighbour][node_to_correct] = amplitude * graph[node_receiving, neighbour]

        for x_node, x_corr in x_corrections.items():
            command_list.append(X(x_node, 0, x_domain=x_corr))

        for z_node, z_corr in z_corrections.items():
            command_list.append(Z(z_node, 0, z_domain=z_corr))

    return Pattern(command_list, input_nodes=graph.input_nodes)
