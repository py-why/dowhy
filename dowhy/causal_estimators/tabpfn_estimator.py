import importlib
import logging
from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.utils import resample

import torch
import multiprocessing as mp

from dowhy.causal_estimator import CausalEstimate, CausalEstimator
from dowhy.causal_estimators.regression_estimator import RegressionEstimator
from dowhy.causal_identifier import IdentifiedEstimand

logger = logging.getLogger(__name__)


def _mp_fit_and_predict_worker(model_type: str, device_id: int, model_kwargs: dict,
                               X_train: np.ndarray, y_train: np.ndarray,
                               X_pred: np.ndarray, want_proba: bool, out_queue):
    """Multiprocessing worker to fit a TabPFN model and predict.

    :param model_type: Model type ("Classifier" or "Regressor")
    :param device_id: CUDA device index for this worker
    :param model_kwargs: Keyword arguments forwarded to TabPFN model
    :param X_train: Feature chunk used to fit the local model
    :param y_train: Outcome values for the local chunk
    :param X_pred: Full feature matrix to predict on
    :param want_proba: If True, return probability for classifiers
    :param out_queue: Multiprocessing queue for results
    """
    try:
        if torch.cuda.is_available():
            torch.cuda.set_device(device_id)
            device = torch.device(f"cuda:{device_id}")
        else:
            device = torch.device("cpu")
        if model_type == "Classifier":
            from tabpfn import TabPFNClassifier as _Model
        else:
            from tabpfn import TabPFNRegressor as _Model
        model = _Model(device=device, **model_kwargs)
        model.fit(X_train, y_train)
        if want_proba and hasattr(model, "predict_proba"):
            pred = model.predict_proba(X_pred)[:, 1]
        else:
            pred = model.predict(X_pred)
        out_queue.put((device_id, pred, None))
    except Exception as e:
        out_queue.put((device_id, None, str(e)))


class TabPFNModelWrapper:
    """Wrapper for TabPFN models with single or multi-GPU support.

    Handles model type resolution, dataset pretraining limits validation,
    and optional multiprocessing-based inference across multiple GPUs.
    """

    def __init__(self, model_type_param: str, model_kwargs: dict, max_num_classes: int = 10, device_ids: Optional[List[int]] = None):
        """Initialize the wrapper with modeling options.

        :param model_type_param: Model type ('classifier', 'regressor', or 'auto')
        :param model_kwargs: Arguments forwarded to TabPFN model
        :param max_num_classes: Threshold for auto-detecting classification tasks
        :param device_ids: List of CUDA device indices for multiprocessing
        """
        self.model_type_param = (model_type_param or "auto").lower()
        self.model_kwargs = dict(model_kwargs)
        self.max_num_classes = int(max_num_classes)
        self.device_ids = list(device_ids or [])

        self.resolved_model_type: Optional[str] = None
        self.train_X: Optional[np.ndarray] = None
        self.train_y: Optional[np.ndarray] = None
        self._single_model = None

    def _resolve_auto(self, outcome_series: pd.Series, logger: logging.Logger) -> str:
        """Automatically determine model type based on outcome characteristics."""
        num_unique = int(pd.Series(outcome_series).nunique())
        dtype = outcome_series.dtype
        looks_categorical = str(dtype) in ["object", "category", "bool"]
        looks_integer = str(dtype) in ["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"]
        is_classifier = looks_categorical or (looks_integer and num_unique <= self.max_num_classes)
        if is_classifier:
            logger.warning(
                f"TabPFN model_type auto-selected: Classifier (dtype={dtype}, unique_classes={num_unique}, max_num_classes={self.max_num_classes})."
            )
            return "Classifier"
        logger.warning(
            f"TabPFN model_type auto-selected: Regressor (dtype={dtype}, unique_classes={num_unique}, max_num_classes={self.max_num_classes})."
        )
        return "Regressor"

    def _advise_pretraining_limits(self, features: np.ndarray, outcome_values: np.ndarray, logger: logging.Logger):
        """Validate dataset against TabPFN pretraining limits."""
        num_samples, num_features = features.shape
        if num_samples > 10000:
            logger.warning("WARNING: TabPFN performs best up to ~10k samples. Your dataset has %d samples.", num_samples)
        if num_features > 500:
            logger.warning("WARNING: TabPFN performs best up to ~500 features. Your dataset has %d features.", num_features)
        if self.resolved_model_type == "Classifier":
            num_classes = int(pd.Series(outcome_values).nunique())
            if num_classes > 10:
                raise ValueError(f"Number of classes {num_classes} exceeds TabPFN limit (10). Reduce classes.")

    def prepare(self, features: np.ndarray, outcome_series: pd.Series, logger: logging.Logger):
        """Prepare the wrapper by resolving model type and validating data.

        :param features: Feature matrix
        :param outcome_series: Outcome column as pandas Series
        :param logger: Logger for warnings
        :returns: self
        """
        if self.model_type_param == "classifier":
            self.resolved_model_type = "Classifier"
        elif self.model_type_param == "regressor":
            self.resolved_model_type = "Regressor"
        else:
            self.resolved_model_type = self._resolve_auto(outcome_series, logger)

        if self.resolved_model_type == "Classifier":
            s = pd.Series(outcome_series)
            unique_vals = pd.unique(s.dropna())
            if s.dtype == bool or set(unique_vals).issubset({0, 1}):
                y = s.astype(np.int64).to_numpy()
            else:
                cats = sorted([str(val) for val in unique_vals])
                y = pd.Categorical(s.astype(str), categories=cats, ordered=True).codes
                y = y.astype(np.int64)
            self.train_y = y
        else:
            self.train_y = outcome_series.to_numpy(dtype=np.float32)
        self.train_X = features

        self._advise_pretraining_limits(self.train_X, self.train_y, logger)
        return self

    def fit_single(self, device: torch.device):
        """Fit a single TabPFN model on the provided device using prepared train_X/train_y."""
        if self.resolved_model_type == "Classifier":
            from tabpfn import TabPFNClassifier as _Model
        else:
            from tabpfn import TabPFNRegressor as _Model
        model = _Model(device=device, **self.model_kwargs)
        model.fit(self.train_X, self.train_y)
        self._single_model = model
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict outcomes for the given features."""
        if self._single_model is not None:
            return self._single_model.predict(features)
        if self.device_ids:
            return self.predict_multiprocess(features, want_proba=False)
        raise ValueError("TabPFNModelWrapper: model not fitted. Call fit_single() or provide device_ids for MP.")

    def predict_proba(self, features: np.ndarray):
        """Return predicted probabilities for classifiers."""
        if self.resolved_model_type != "Classifier":
            return None
        if self._single_model is not None:
            if hasattr(self._single_model, "predict_proba"):
                return self._single_model.predict_proba(features)
            return None
        if self.device_ids:
            return self.predict_multiprocess(features, want_proba=True)
        return None

    def predict_multiprocess(self, features: np.ndarray, want_proba: bool) -> np.ndarray:
        """Fit per-device models and average predictions across devices.

        :param features: Feature matrix to predict on (shared across workers)
        :param want_proba: If True, return probabilities for classifiers
        :returns: Averaged predictions
        """
        if not self.device_ids:
            raise ValueError("No device_ids provided for multiprocessing path.")

        num_devices = len(self.device_ids)
        num_samples = self.train_X.shape[0]
        chunk_size = max(1, num_samples // num_devices)
        out_queue = mp.Queue()
        procs: List[mp.Process] = []
        
        for i, device_id in enumerate(self.device_ids):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size if i < num_devices - 1 else num_samples
            X_chunk = self.train_X[start_idx:end_idx]
            y_chunk = self.train_y[start_idx:end_idx]
            p = mp.Process(
                target=_mp_fit_and_predict_worker,
                args=(self.resolved_model_type, device_id, self.model_kwargs, X_chunk, y_chunk, features, want_proba, out_queue),
            )
            p.start()
            procs.append(p)
        
        results = []
        for _ in procs:
            device_id, pred, err = out_queue.get()
            results.append((device_id, pred, err))
        
        for p in procs:
            p.join()
        
        preds = []
        errors = []
        for device_id, pred, err in results:
            if err is not None:
                errors.append((device_id, err))
            else:
                preds.append(pred)
        
        if errors:
            raise RuntimeError(f"TabPFN multiprocessing prediction errors: {errors}")
        return np.mean(preds, axis=0)


class TabpfnEstimator(RegressionEstimator):
    def __init__(
        self,
        identified_estimand: IdentifiedEstimand,
        test_significance: Union[bool, str] = False,
        evaluate_effect_strength: bool = False,
        confidence_intervals: bool = False,
        num_null_simulations: int = CausalEstimator.DEFAULT_NUMBER_OF_SIMULATIONS_STAT_TEST,
        num_simulations: int = CausalEstimator.DEFAULT_NUMBER_OF_SIMULATIONS_CI,
        sample_size_fraction: int = CausalEstimator.DEFAULT_SAMPLE_SIZE_FRACTION,
        confidence_level: float = CausalEstimator.DEFAULT_CONFIDENCE_LEVEL,
        need_conditional_estimates: Union[bool, str] = "auto",
        num_quantiles_to_discretize_cont_cols: int = CausalEstimator.NUM_QUANTILES_TO_DISCRETIZE_CONT_COLS,
        method_params: Optional[dict] = None,
        **kwargs,
    ):
        """
        :param identified_estimand: probability expression
            representing the target identified estimand to estimate.
        :param test_significance: Binary flag or a string indicating whether to test significance and by which method. All estimators support test_significance="bootstrap" that estimates a p-value for the obtained estimate using the bootstrap method. Individual estimators can override this to support custom testing methods. The bootstrap method supports an optional parameter, num_null_simulations. If False, no testing is done. If True, significance of the estimate is tested using the custom method if available, otherwise by bootstrap.
        :param evaluate_effect_strength: (Experimental) whether to evaluate the strength of effect
        :param confidence_intervals: Binary flag or a string indicating whether the confidence intervals should be computed and which method should be used. All methods support estimation of confidence intervals using the bootstrap method by using the parameter confidence_intervals="bootstrap". The bootstrap method takes in two arguments (num_simulations and sample_size_fraction) that can be optionally specified in the params dictionary. Estimators may also override this to implement their own confidence interval method. If this parameter is False, no confidence intervals are computed. If True, confidence intervals are computed by the estimator's specific method if available, otherwise through bootstrap
        :param num_null_simulations: The number of simulations for testing the
            statistical significance of the estimator
        :param num_simulations: The number of simulations for finding the
            confidence interval (and/or standard error) for a estimate
        :param sample_size_fraction: The size of the sample for the bootstrap
            estimator
        :param confidence_level: The confidence level of the confidence
            interval estimate
        :param need_conditional_estimates: Boolean flag indicating whether
            conditional estimates should be computed. Defaults to True if
            there are effect modifiers in the graph
        :param num_quantiles_to_discretize_cont_cols: The number of quantiles
            into which a numeric effect modifier is split, to enable
            estimation of conditional treatment effect over it.
        :param method_params: Dictionary of TabPFN-specific parameters
            - model_type: "auto", "classifier", or "regressor" (default: "auto")
            - n_estimators: Number of TabPFN models to ensemble (default: 8)
            - max_num_classes: Threshold for classification task detection (default: 10)
            - use_multi_gpu: Whether to use multiple GPUs (default: False)
            - device_ids: List of GPU device IDs (default: [])
        :param kwargs: (optional) Additional estimator-specific parameters
        """
        # Extract TabPFN-specific params from kwargs and merge into method_params
        # to support both method_params={...} and **method_params call patterns
        tabpfn_keys = {"model_type", "n_estimators", "max_num_classes", "use_multi_gpu", "device_ids"}
        tabpfn_params = {k: v for k, v in kwargs.items() if k in tabpfn_keys}
        other_kwargs = {k: v for k, v in kwargs.items() if k not in tabpfn_keys}
        self.method_params = {**(method_params or {}), **tabpfn_params}

        super().__init__(
            identified_estimand=identified_estimand,
            test_significance=test_significance,
            evaluate_effect_strength=evaluate_effect_strength,
            confidence_intervals=confidence_intervals,
            num_null_simulations=num_null_simulations,
            num_simulations=num_simulations,
            sample_size_fraction=sample_size_fraction,
            confidence_level=confidence_level,
            need_conditional_estimates=need_conditional_estimates,
            num_quantiles_to_discretize_cont_cols=num_quantiles_to_discretize_cont_cols,
            **other_kwargs,
        )
        self.logger.info("INFO: Using TabPFN Estimator")
        self.tabpfn_model = None
        self._use_multi_gpu = bool(self.method_params.get("use_multi_gpu", False))
        self.device_ids = self.method_params.get("device_ids", [])
        
        self._check_tabpfn_dependencies()
        
        if self._use_multi_gpu and not self.device_ids and torch.cuda.is_available():
            self.device_ids = list(range(torch.cuda.device_count()))
            self.logger.info(f"INFO: Auto-detected {len(self.device_ids)} GPUs: {self.device_ids}")
        
        self._device = self._get_device()

    def _check_tabpfn_dependencies(self):
        """Checks if TabPFN and PyTorch are installed."""
        try:
            importlib.import_module("tabpfn")
            importlib.import_module("torch")
        except ImportError:
            raise ImportError(
                "TabPFNEstimator requires tabpfn and torch to be installed. "
                "Please install them with: pip install tabpfn torch\n\n"
                "Note: TabPFN v2.5 models require HuggingFace authentication.\n"
                "If you encounter authentication errors:\n"
                "  1. Visit https://huggingface.co/Prior-Labs/tabpfn_2_5 and accept terms\n"
                "  2. Run: huggingface-cli login\n"
                "  3. Or set environment variable: export HF_TOKEN=\"your_huggingface_token\"\n"
                "For more details: https://docs.priorlabs.ai/how-to-access-gated-models"
            )

    def _get_device(self):
        """Determines the appropriate PyTorch device (GPU or CPU) to use."""
        if torch.cuda.is_available():
            device = "cuda"
            self.logger.info("INFO: Using GPU for TabPFN. Make sure CUDA is enabled.")
        else:
            device = "cpu"
            self.logger.warning(
                "WARNING: No GPU found. TabPFN will run on CPU, which can be very slow."
                "For better performance, consider using a GPU instance."
            )
        
        return torch.device(device)

    def fit(
        self,
        data: pd.DataFrame,
        effect_modifier_names: Optional[List[str]] = None,
    ):
        """
        Fits the estimator with data for effect estimation
        :param data: data frame containing the data
        :param treatment: name of the treatment variable
        :param outcome: name of the outcome variable
        :param effect_modifiers: Variables on which to compute separate
                    effects, or return a heterogeneous effect function. Not all
                    methods support this currently.
        """
        self.reset_encoders()
        self._set_effect_modifiers(data, effect_modifier_names)

        self.logger.debug("Adjustment set variables used:" + ",".join(self._target_estimand.get_adjustment_set()))
        self._observed_common_causes_names = self._target_estimand.get_adjustment_set()
        if len(self._observed_common_causes_names) > 0:
            self._observed_common_causes = data[self._observed_common_causes_names]
            self._observed_common_causes = self._encode(self._observed_common_causes, "observed_common_causes")
        else:
            self._observed_common_causes = None

        self.symbolic_estimator = self.construct_symbolic_estimator(self._target_estimand)
        self.logger.info(self.symbolic_estimator)

        _, self.model = self._build_model(data)
        return self

    def estimate_effect(
        self,
        data: pd.DataFrame = None,
        treatment_value: Any = 1,
        control_value: Any = 0,
        target_units=None,
        need_conditional_estimates: bool = None,
        **_,
    ):
        self._target_units = target_units
        self._treatment_value = treatment_value
        self._control_value = control_value

        if need_conditional_estimates is None:
            need_conditional_estimates = self.need_conditional_estimates

        effect_estimate = self._do(treatment_value, data) - self._do(control_value, data)
        conditional_effect_estimates = None
        if need_conditional_estimates:
            conditional_effect_estimates = self._estimate_conditional_effects(
                data, self._estimate_effect_fn, effect_modifier_names=self._effect_modifier_names
            )

        effect_intervals = None
        if self._confidence_intervals:
            effect_intervals = self._estimate_confidence_intervals_with_bootstrap(
                data, 
                effect_estimate, 
                confidence_level=self.confidence_level,
                num_simulations=self.num_simulations,
                sample_size_fraction=self.sample_size_fraction
            )

        estimate = CausalEstimate(
            data=data,
            treatment_name=self._target_estimand.treatment_variable,
            outcome_name=self._target_estimand.outcome_variable,
            estimate=effect_estimate,
            control_value=control_value,
            treatment_value=treatment_value,
            conditional_estimates=conditional_effect_estimates,
            target_estimand=self._target_estimand,
            realized_estimand_expr=self.symbolic_estimator,
            effect_intervals=effect_intervals,
        )

        estimate.add_estimator(self)
        return estimate
    
    def _build_model(self, data: pd.DataFrame):
        """Build and fit TabPFN model wrapper.

        :param data: DataFrame containing treatment, outcome and confounders
        :returns: Tuple of (features, model wrapper)
        """
        features = self._build_features(data)
        tabpfn_features = features[:, 1:]  # remove intercept column for TabPFN
        outcome_col = self._target_estimand.outcome_variable[0]
        outcome_series = data[outcome_col]

        wrapper = TabPFNModelWrapper(
            model_type_param=self.method_params.get("model_type", "auto"),
            model_kwargs={"n_estimators": self.method_params.get("n_estimators", 8)},
            max_num_classes=self.method_params.get("max_num_classes", 10),
            device_ids=self.device_ids,
        )
        wrapper.prepare(tabpfn_features, outcome_series, self.logger)
        
        if not wrapper.device_ids:
            wrapper.fit_single(self._device)
        
        self.tabpfn_model = wrapper
        return (features, wrapper)
    
    def predict_fn(self, data: pd.DataFrame, model, features):
        tabpfn_features = features[:, 1:]
        if getattr(model, "resolved_model_type", None) == "Classifier":
            proba = model.predict_proba(tabpfn_features)
            if proba is not None:
                return proba[:, 1] if proba.ndim == 2 else proba
            return model.predict(tabpfn_features)
        return model.predict(tabpfn_features)

    def _generate_bootstrap_estimates(self, data, num_bootstrap_simulations, sample_size_fraction):
        simulation_results = np.zeros(num_bootstrap_simulations)
        sample_size = int(sample_size_fraction * len(data))
        
        self.logger.info(f"TabPFN Bootstrap: {num_bootstrap_simulations} simulations, sample_size: {sample_size}")
        
        for index in range(num_bootstrap_simulations):
            new_data = resample(data, n_samples=sample_size)
            new_estimator = self.get_new_estimator_object(
                self._target_estimand,
                test_significance=False,
                evaluate_effect_strength=False,
                confidence_intervals=False,
            )
            new_estimator.fit(new_data, effect_modifier_names=self._effect_modifier_names)
            new_effect = new_estimator.estimate_effect(
                new_data,
                treatment_value=self._treatment_value,
                control_value=self._control_value,
                target_units=self._target_units,
            )
            simulation_results[index] = new_effect.value
        
        return CausalEstimator.BootstrapEstimates(
            simulation_results,
            {"num_simulations": num_bootstrap_simulations, "sample_size_fraction": sample_size_fraction}
        )

    def construct_symbolic_estimator(self, estimand):
        expr = "E[{} | do({})] = E[{} | {}, {}]".format(
            self._target_estimand.outcome_variable[0],
            self._target_estimand.treatment_variable[0],
            self._target_estimand.outcome_variable[0],
            ", ".join(self._target_estimand.treatment_variable),
            ", ".join(self._target_estimand.get_backdoor_variables()),
        )
        return expr
    