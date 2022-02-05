import pytest
import numpy as np
from scipy import stats

from dowhy.causal_refuter import CausalRefuter
from dowhy.causal_identifier import IdentifiedEstimand
from dowhy.causal_model import CausalModel
from dowhy.causal_estimator import CausalEstimate
from dowhy.datasets import linear_dataset


class MockRefuter(CausalRefuter):
	pass


def test_causal_refuter_placeholder_method():
	refuter = MockRefuter(None, IdentifiedEstimand(None, None, None), None)
	with pytest.raises(NotImplementedError):
		refuter.refute_estimate()


def test_choose_variables():
	data = linear_dataset(1, num_common_causes=2, num_instruments=1, num_samples=100)
	model = CausalModel(data["df"], treatment=data["treatment_name"], outcome=data["outcome_name"], graph=data["gml_graph"])
	estimand = model.identify_effect()
	estimate = model.estimate_effect(estimand)
	refuter = CausalRefuter(data["df"], estimand, estimate)

	assert set(refuter.choose_variables(True)) == {"W0", "W1", "Z0"}
	assert refuter.choose_variables(False) == []
	assert set(refuter.choose_variables(["W0", "W1", "Z0"])) == {"W0", "W1", "Z0"}
	assert set(refuter.choose_variables(("W0", "W1", "Z0"))) == {"W0", "W1", "Z0"}
	assert set(refuter.choose_variables(["-W0", "-Z0"])) == {"W1"}
	assert refuter.choose_variables(["-W0", "-W1", "-Z0"]) == []

	with pytest.raises(ValueError):
		refuter.choose_variables(["W0", "-W1"])
	
	with pytest.raises(ValueError):
		refuter.choose_variables(["W0", "treat"])

	with pytest.raises(TypeError):
		refuter.choose_variables("W0")


def test_test_significance():
	data = stats.norm.isf(np.linspace(0, 1, 102)[1:-1])
	refuter = CausalRefuter(None, IdentifiedEstimand(None, None, None), None)

	make_estimate = lambda v: CausalEstimate(v, None, None, None, None)

	assert not refuter.test_significance(make_estimate(data[50]), data, "bootstrap", significance_level=0.1)["is_statistically_significant"]
	assert not refuter.test_significance(make_estimate(data[5]), data, "bootstrap", significance_level=0.1)["is_statistically_significant"]
	assert not refuter.test_significance(make_estimate(data[94]), data, "bootstrap", significance_level=0.1)["is_statistically_significant"]
	assert refuter.test_significance(make_estimate(data[3]), data, "bootstrap", significance_level=0.1)["is_statistically_significant"]
	assert refuter.test_significance(make_estimate(data[96]), data, "bootstrap", significance_level=0.1)["is_statistically_significant"]

	assert not refuter.test_significance(make_estimate(data[50]), data, "normal_test", significance_level=0.1)["is_statistically_significant"]
	assert not refuter.test_significance(make_estimate(data[5]), data, "normal_test", significance_level=0.1)["is_statistically_significant"]
	assert not refuter.test_significance(make_estimate(data[94]), data, "normal_test", significance_level=0.1)["is_statistically_significant"]
	assert refuter.test_significance(make_estimate(data[3]), data, "normal_test", significance_level=0.1)["is_statistically_significant"]
	assert refuter.test_significance(make_estimate(data[96]), data, "normal_test", significance_level=0.1)["is_statistically_significant"]
