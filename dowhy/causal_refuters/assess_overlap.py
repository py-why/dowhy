import logging
from typing import List

from dowhy.causal_refuter import CausalRefuter
from dowhy.causal_refuters.assess_overlap_overrule import OverruleAnalyzer

logger = logging.getLogger(__name__)


class AssessOverlap(CausalRefuter):
    """Assess Overlap

    AssessOverlap class implements the OverRule method from Oberst et al. 2020
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the parameters required for the refuter.

        This is called with arguments passed through `refute_estimate`

        :param: cat_feats: List[str]: List of categorical features, all
        others will be discretized
        """
        super().__init__(*args, **kwargs)
        self._backdoor_vars = self._target_estimand.get_backdoor_variables()
        self._cat_feats = kwargs.pop("cat_feats", [])
        self._support_config = kwargs.pop("support_config", None)
        self._overlap_config = kwargs.pop("overlap_config", None)
        self._overlap_eps = kwargs.pop("overlap_eps", 0.1)
        self._overrule_verbose = kwargs.pop("overrule_verbose", False)

    def refute_estimate(self, show_progress_bar=False):
        return assess_support_and_overlap_overrule(
            data=self._data,
            backdoor_vars=self._backdoor_vars,
            treatment_name=self._treatment_name,
            cat_feats=self._cat_feats,
            overlap_config=self._overlap_config,
            support_config=self._support_config,
            overlap_eps=self._overlap_eps,
            verbose=self._overrule_verbose,
        )


def assess_support_and_overlap_overrule(
    data, backdoor_vars, treatment_name, cat_feats, overlap_config, support_config, overlap_eps, verbose
):
    X = data[backdoor_vars]
    t = data[treatment_name]
    analyzer = OverruleAnalyzer(
        cat_feats=cat_feats,
        overlap_config=overlap_config,
        support_config=support_config,
        overlap_eps=overlap_eps,
        verbose=verbose,
    )
    analyzer.fit(X, t)
    return analyzer
