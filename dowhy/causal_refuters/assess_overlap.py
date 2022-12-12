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
        if "cat_feats" in kwargs:
            self._cat_feats = kwargs.get("cat_feats")
        else:
            self._cat_feats = []

    def refute_estimate(self, show_progress_bar=False):
        return assess_support_and_overlap_overrule(
            data=self._data,
            backdoor_vars=self._backdoor_vars,
            treatment_name=self._treatment_name,
            cat_feats=self._cat_feats,
        )


def assess_support_and_overlap_overrule(data, backdoor_vars, treatment_name, cat_feats: List[str] = []):
    X = data[backdoor_vars]
    t = data[treatment_name]
    analyzer = OverruleAnalyzer(cat_feats=cat_feats)
    analyzer.fit(X, t)
    return analyzer
