import pytest

from dowhy.causal_refuter import CausalRefuter
from dowhy.causal_identifier import IdentifiedEstimand


class MockRefuter(CausalRefuter):
	pass


def test_causal_refuter_placeholder_method():
	refuter = MockRefuter(None, IdentifiedEstimand(None, None, None), None)
	with pytest.raises(NotImplementedError):
		refuter.refute_estimate()
