import pandas as pd

from dowhy.interpreters.textual_interpreter import TextualInterpreter


class TextualEffectInterpreter(TextualInterpreter):

    SUPPORTED_ESTIMATORS = ["all"]

    def __init__(self, instance, **kwargs):
        super().__init__(instance, **kwargs)
        # Setting estimator attribute for convenience
        self.estimator = self.estimate.estimator

    def interpret(self, data: pd.DataFrame):
        """Interpret causal effect by showing how much a unit change in treatment will cause change in the outcome."""
        interpret_text = ""
        treatments_str = ",".join(self.estimate._treatment_name)
        if pd.api.types.is_numeric_dtype(data[self.estimate._outcome_name].dtypes):
            # Outcome is numeric
            if all(
                pd.api.types.is_numeric_dtype(tr_dtype) or pd.api.types.is_bool_dtype(tr_dtype)
                for tr_dtype in data[self.estimate._treatment_name].dtypes
            ):
                # Treatments are also numeric or binary
                interpret_text += "Increasing the treatment variable(s) [{0}] from {1} to {2} causes an increase of {3} in the expected value of the outcome [{4}]".format(
                    treatments_str,
                    self.estimator._control_value,
                    self.estimator._treatment_value,
                    self.estimate.value,
                    self.estimate._outcome_name,
                )
            else:
                raise NotImplementedError("Interpretation not supported yet for categorical treatments")
        else:
            # Outcome is categorical
            if all(
                pd.api.types.is_numeric_dtype(tr_dtype) or pd.api.types.is_bool_dtype(tr_dtype)
                for tr_dtype in data[self.estimate._treatment_name].dtypes
            ):
                # Treatments are numeric or binary
                interpret_text += "Increasing the treatment variable(s) [{0}] from {1} to {2} causes an increase of {3} in the expected value of the outcome [{4}]".format(
                    treatments_str,
                    self.estimator._control_value,
                    self.estimator._treatment_value,
                    self.estimate.value,
                    self.estimate._outcome_name,
                )
            else:
                raise NotImplementedError("Interpretation not supported yet for categorical treatments")

        interpret_text += ", over the data distribution/population represented by the dataset."
        self.show(interpret_text)
        return interpret_text
