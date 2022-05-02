
"""This module defines multiple implementations of the abstract class
:py:class:`FunctionalCausalModel <causality.graph.interface.FunctionalCausalModel>` (FCM)
"""

import copy
from abc import abstractmethod, ABC
from typing import Optional, List

import numpy as np

from dowhy.gcm.graph import StochasticModel, FunctionalCausalModel, InvertibleFunctionalCausalModel
from dowhy.gcm.util.general import shape_into_2d, is_categorical


class PredictionModel:
    """ Represents general prediction model implementations. Each prediction model should provide a fit and a predict
        method. """

    @abstractmethod
    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def clone(self):
        """
        Clones the prediction model using the same hyper parameters but not fitted.

        :return: An unfitted clone of the prediction model.
        """
        raise NotImplementedError


class ClassificationModel(PredictionModel):

    @abstractmethod
    def predict_probabilities(self, X: np.array) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def classes(self) -> List[str]:
        raise NotImplementedError


class InvertibleFunction:

    @abstractmethod
    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """ Applies the function on the input. """
        raise NotImplementedError

    @abstractmethod
    def evaluate_inverse(self, X: np.ndarray) -> np.ndarray:
        """ Returns the outcome of applying the inverse of the function on the inputs. """
        raise NotImplementedError


class PostNonlinearModel(InvertibleFunctionalCausalModel):
    """
        Represents an post nonlinear FCM, i.e. models of the form:
            Y = g(f(X) + N),
        where X are parent nodes of the target node Y, f an arbitrary prediction model expecting inputs from the
        parents X, N a noise variable and g an invertible function.
    """

    def __init__(self,
                 prediction_model: PredictionModel,
                 noise_model: StochasticModel,
                 invertible_function: InvertibleFunction) -> None:
        """
        :param prediction_model: The prediction model f.
        :param invertible_function: The invertible function g.
        :param noise_model: The StochasticModel to describe the distribution of the noise N.
        """
        self.__prediction_model = prediction_model
        self.__noise_model = noise_model
        self.__invertible_function = invertible_function

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        X, Y = shape_into_2d(X, Y)

        self.__prediction_model.fit(X=X, Y=self.__invertible_function.evaluate_inverse(Y))
        self.__noise_model.fit(X=self.estimate_noise(Y, X))

    def estimate_noise(self,
                       target_samples: np.ndarray,
                       parent_samples: np.ndarray) -> np.ndarray:
        target_samples, parent_samples = shape_into_2d(target_samples, parent_samples)

        return self.__invertible_function.evaluate_inverse(target_samples) - self.__prediction_model.predict(
            parent_samples)

    def draw_noise_samples(self, num_samples: int) -> np.ndarray:
        return self.__noise_model.draw_samples(num_samples)

    def evaluate(self, parent_samples: np.ndarray, noise_samples: np.ndarray) -> np.ndarray:
        parent_samples, noise_samples = shape_into_2d(parent_samples, noise_samples)
        predictions = shape_into_2d(self.__prediction_model.predict(parent_samples))

        return self.__invertible_function.evaluate(predictions + noise_samples)

    def __str__(self) -> str:
        return '%s with %s and an %s' % (self.__class__.__name__,
                                         self.__prediction_model.__class__.__name__,
                                         self.__invertible_function.__class__.__name__)

    def clone(self):
        return PostNonlinearModel(prediction_model=self.__prediction_model.clone(),
                                  noise_model=self.__noise_model.clone(),
                                  invertible_function=copy.deepcopy(self.__invertible_function))

    @property
    def prediction_model(self) -> PredictionModel:
        return self.__prediction_model

    @property
    def noise_model(self) -> StochasticModel:
        return self.__noise_model

    @property
    def invertible_function(self) -> InvertibleFunction:
        return self.__invertible_function


class AdditiveNoiseModel(PostNonlinearModel):
    def __init__(self,
                 prediction_model: PredictionModel,
                 noise_model: Optional[StochasticModel] = None) -> None:
        if noise_model is None:
            from dowhy.gcm.stochastic_models import EmpiricalDistribution
            noise_model = EmpiricalDistribution()

        from dowhy.gcm.ml.regression import InvertibleIdentityFunction
        super(AdditiveNoiseModel, self).__init__(prediction_model=prediction_model,
                                                 noise_model=noise_model,
                                                 invertible_function=InvertibleIdentityFunction())

    def clone(self):
        return AdditiveNoiseModel(prediction_model=self.prediction_model.clone(),
                                  noise_model=self.noise_model.clone())


class ProbabilityEstimatorModel(ABC):
    @abstractmethod
    def estimate_probabilities(self, parent_samples: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class ClassifierFCM(FunctionalCausalModel, ProbabilityEstimatorModel):
    def __init__(self, classifier_model: Optional[ClassificationModel] = None) -> None:
        self.__classifier_model = classifier_model

        if classifier_model is None:
            from dowhy.gcm.ml.classification import create_hist_gradient_boost_classifier
            self.__classifier_model = create_hist_gradient_boost_classifier()

    def draw_noise_samples(self, num_samples: int) -> np.ndarray:
        return shape_into_2d(np.random.uniform(0, 1, num_samples))

    def evaluate(self, parent_samples: np.ndarray, noise_samples: np.ndarray) -> np.ndarray:
        noise_samples = shape_into_2d(noise_samples)
        probabilities = self.estimate_probabilities(parent_samples)

        probabilities = np.cumsum(probabilities, axis=1) - noise_samples
        probabilities[probabilities < 0] = 1

        return shape_into_2d(np.array(self.get_class_names(np.argmin(probabilities, axis=1))))

    def estimate_probabilities(self, parent_samples: np.ndarray) -> np.ndarray:
        return self.__classifier_model.predict_probabilities(parent_samples)

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        X, Y = shape_into_2d(X, Y)

        if not is_categorical(Y):
            raise ValueError("The target data needs to be categorical in the form of strings!")

        self.__classifier_model.fit(X=X, Y=Y)

    def clone(self):
        return ClassifierFCM(classifier_model=self.__classifier_model.clone())

    def get_class_names(self, class_indices: np.ndarray) -> List[str]:
        return [self.__classifier_model.classes()[index] for index in class_indices]

    @property
    def classifier_model(self) -> ClassificationModel:
        return self.__classifier_model
