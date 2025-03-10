import logging
import warnings
from typing import List, Optional

from dowhy.causal_refuter import CausalRefuter
from dowhy.causal_refuters.assess_overlap_overrule import OverlapConfig, OverruleAnalyzer, SupportConfig

logger = logging.getLogger(__name__)


class AssessOverlap(CausalRefuter):
    """Assess Overlap

    This class implements the OverRule algorithm for assessing support and overlap via Boolean Rulesets, from [1].

    [1] Oberst, M., Johansson, F., Wei, D., Gao, T., Brat, G., Sontag, D., & Varshney, K. (2020). Characterization of
    Overlap in Observational Studies. In S. Chiappa & R. Calandra (Eds.), Proceedings of the Twenty Third International
    Conference on Artificial Intelligence and Statistics (Vol. 108, pp. 788–798). PMLR. https://arxiv.org/abs/1907.04138
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the parameters required for the refuter.

        Arguments are passed through to the `refute_estimate` method. See dowhy.causal_refuters.assess_overlap_overrule
        for the definition of the `SupportConfig` and `OverlapConfig` dataclasses that define optimization
        hyperparameters.

        .. warning::
            This method is only compatible with estimators that use backdoor adjustment, and will attempt to acquire
            the set of backdoor variables via `self._target_estimand.get_backdoor_variables()`.

        :param: cat_feats: List[str]: List of categorical features, all others will be discretized
        :param: support_config: SupportConfig: DataClass with configuration options for learning support rules
        :param: overlap_config: OverlapConfig: DataClass with configuration options for learning overlap rules
        :param: overlap_eps: float: Defines the range of propensity scores for a point to be considered in the overlap
            region, with the range defined as `(overlap_eps, 1 - overlap_eps)`, defaults to 0.1
        :param: overrule_verbose: bool: Enable verbose logging of optimization output, defaults to False
        :param: support_only: bool: Only fit rules to describe the support region (do not fit overlap rules), defaults to False
        :param: overlap_only: bool: Only fit rules to describe the overlap region (do not fit support rules), defaults to False
        """
        super().__init__(*args, **kwargs)
        # TODO: Check that the target estimand has backdoor variables?
        # TODO: Add support for the general adjustment criterion.
        self._backdoor_vars = self._target_estimand.get_backdoor_variables()
        self._cat_feats = kwargs.pop("cat_feats", [])
        self._support_config = kwargs.pop("support_config", None)
        self._overlap_config = kwargs.pop("overlap_config", None)
        self._overlap_eps = kwargs.pop("overlap_eps", 0.1)
        if self._overlap_eps < 0 or self._overlap_eps > 1:
            raise ValueError(f"Value of `overlap_eps` must be in [0, 1], got {self._overlap_eps}")
        self._support_only = kwargs.pop("support_only", False)
        self._overlap_only = kwargs.pop("overlap_only", False)
        self._overrule_verbose = kwargs.pop("overrule_verbose", False)

    def refute_estimate(self, show_progress_bar=False):
        """
        Learn overlap and support rules.

        :param show_progress_bar: Not implemented, will raise error if set to True, defaults to False
        :type show_progress_bar: bool
        :raises NotImplementedError: Will raise this error if show_progress_bar=True
        :returns: object of class OverruleAnalyzer
        """
        if show_progress_bar:
            warnings.warn("No progress bar is available for OverRule")

        return assess_support_and_overlap_overrule(
            data=self._data,
            backdoor_vars=self._backdoor_vars,
            treatment_name=self._treatment_name,
            cat_feats=self._cat_feats,
            overlap_config=self._overlap_config,
            support_config=self._support_config,
            overlap_eps=self._overlap_eps,
            support_only=self._support_only,
            overlap_only=self._overlap_only,
            verbose=self._overrule_verbose,
        )


def assess_support_and_overlap_overrule(
    data,
    backdoor_vars: List[str],
    treatment_name: str,
    cat_feats: List[str] = [],
    overlap_config: Optional[OverlapConfig] = None,
    support_config: Optional[SupportConfig] = None,
    overlap_eps: float = 0.1,
    support_only: bool = False,
    overlap_only: bool = False,
    verbose: bool = False,
):
    """
    Learn support and overlap rules using OverRule.

    :param data: Data containing backdoor variables and treatment name
    :param backdoor_vars: List of backdoor variables. Support and overlap rules will only be learned with respect to
    these variables
    :type backdoor_vars: List[str]
    :param treatment_name: Treatment name
    :type treatment_name: str
    :param cat_feats: Categorical features
    :type cat_feats: List[str]
    :param overlap_config: Configuration for learning overlap rules
    :type overlap_config: OverlapConfig
    :param support_config: Configuration for learning support rules
    :type support_config: SupportConfig
    :param: overlap_eps: float: Defines the range of propensity scores for a point to be considered in the overlap
        region, with the range defined as `(overlap_eps, 1 - overlap_eps)`, defaults to 0.1
    :param: support_only: bool: Only fit the support region
    :param: overlap_only: bool: Only fit the overlap region
    :param: verbose: bool: Enable verbose logging of optimization output, defaults to False
    """
    analyzer = OverruleAnalyzer(
        backdoor_vars=backdoor_vars,
        treatment_name=treatment_name,
        cat_feats=cat_feats,
        overlap_config=overlap_config,
        support_config=support_config,
        overlap_eps=overlap_eps,
        support_only=support_only,
        overlap_only=overlap_only,
        verbose=verbose,
    )
    analyzer.fit(data)
    return analyzer
