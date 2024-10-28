import logging

import dowhy.causal_estimator
import dowhy.causal_model
import dowhy.causal_refuter


class Interpreter:
    """Base class for all interpretation methods."""

    # Can use these lists to specify the models/estimators/refuters that a particular interpreter supports.  Throw a ValueError if the user provides an incompatible object to intepret.
    SUPPORTED_MODELS = []
    SUPPORTED_ESTIMATORS = []
    SUPPORTED_REFUTERS = []

    def __init__(self, instance, **kwargs):
        """Initialize an interpreter.

        :param instance: An object of type CausalModel, CausalEstimate or CausalRefutation.

        """

        self.model = None
        self.estimate = None
        self.refutation = None

        if isinstance(instance, dowhy.causal_model.CausalModel):
            self.model = instance
        elif isinstance(instance, dowhy.causal_estimator.CausalEstimate):
            self.estimate = instance
        elif isinstance(instance, dowhy.causal_refuter.CausalRefutation):
            self.refutation = instance
        else:
            self.logger.error("Type of object passed not supported for interpretation.")

        # Unpacking the keyword arguments
        if kwargs is not None:
            for key, value in kwargs.items():
                setattr(self, key, value)

        self.logger = logging.getLogger(__name__)

    def interpret(self):
        """Method that implements the functionality of an interpreter.

        To be overridden by interpreter sub-classes.
        """
        raise NotImplementedError
