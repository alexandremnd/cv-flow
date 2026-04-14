from cvflow.command import Command

class Pattern:
    """A pattern is a sequence of commands applied to a graph state.

    Attributes:
        commands (list[Command]): The list of commands in the pattern.
    """
    def __init__(self, commands: list[Command]):
        self.commands = commands

    def check_validity(self):
        """Check the validity of the pattern.

        This method checks that the commands in the pattern are valid, i.e., that they can be applied to a graph state without contradictions.
        For example, it checks that a preparation command (N) is not applied on a node that has not been yet measured.

        """
        raise NotImplementedError("Pattern validity checking is not implemented yet.")

    def reorder_normal(self):
        """Reorder the commands in the pattern to normal form.

        The normal form is defined as follows:
            - All preparation commands (N) are applied first.
            - All entanglement commands (E) are applied second.
            - All measurement commands (M) are applied third.
            - All correction commands (X and Z) are applied last.
        """

        raise NotImplementedError("Reordering to normal form is not implemented yet.")

    def __str__(self):
        num_commands = len(self.commands)
        width = len(str(num_commands))
        return "\n".join(f"{i+1:0{width}d}) {cmd}" for i, cmd in enumerate(self.commands))