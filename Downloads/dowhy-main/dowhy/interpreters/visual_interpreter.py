from dowhy.interpreter import Interpreter


class VisualInterpreter(Interpreter):
    """Base class for interpreters that show plots or visualizations as output."""

    def __init__(self, instance, **kwargs):
        super().__init__(instance, **kwargs)

    def show(self, interpret_plot):
        """Display the intepretation.

        :param interpret_plot: Plot object containing the interpretation

        :returns: None

        """
        # TODO: A common way to show all plots
        raise NotImplementedError
