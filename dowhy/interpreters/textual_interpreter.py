from dowhy.interpreter import Interpreter


class TextualInterpreter(Interpreter):
    """Base class for interpreters that show text as output."""

    def __init__(self, instance, **kwargs):
        super().__init__(instance, **kwargs)

    def show(self, interpret_text):
        """Display the interpretation.

        :param interpret_text: String containing the interpretation

        :returns: None

        """
        print(interpret_text)  # can be extended later to provide a prettier output
