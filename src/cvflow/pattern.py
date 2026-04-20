from cvflow.command import Command, CommandKind

class Pattern:
    """A sequence of commands applied to a graph state.

    Attributes
    ----------
    commands : list[Command]
        The list of commands in the pattern.
    """
    def __init__(self, commands: list[Command]):
        self.commands = commands

        self.check_validity()

    def check_validity(self):
        """Check the validity of the pattern.

        Verifies that the commands can be applied to a graph state without
        contradictions, for example, that a preparation command (N) is not
        applied to a node that has already been measured.

        Raises
        ------
        NotImplementedError
            This method is not yet implemented.
        """
        for i, cmd in enumerate(self.commands):
            if cmd.kind == CommandKind.N:
                for next_cmd in self.commands[i+1:]:
                    is_not_preparation = next_cmd.kind != CommandKind.N
                    is_same_mode = next_cmd.node == cmd.node if next_cmd.kind != CommandKind.E else cmd.node in next_cmd.nodes

                    if is_not_preparation and is_same_mode:
                        raise ValueError(f"Node {cmd.node} cannot be modified by {next_cmd} before being prepared.")


        raise NotImplementedError("Pattern validity checking is not implemented yet.")

    def reorder_normal(self):
        """Reorder the commands into normal form.

        The normal form is defined as follows:

        1. All preparation commands (N) first.
        2. All entanglement commands (E) second.
        3. All measurement commands (M) third.
        4. All correction commands (X and Z) last.

        Raises
        ------
        NotImplementedError
            This method is not yet implemented.
        """
        raise NotImplementedError("Reordering to normal form is not implemented yet.")

    def __str__(self):
        num_commands = len(self.commands)
        width = len(str(num_commands))
        return "\n".join(f"{i+1:0{width}d}) {cmd}" for i, cmd in enumerate(self.commands))
