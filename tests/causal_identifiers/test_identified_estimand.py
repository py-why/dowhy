"""Tests for IdentifiedEstimand, covering edge-cases in get_backdoor_variables."""

from dowhy.causal_identifier.identified_estimand import IdentifiedEstimand


class TestGetBackdoorVariables:
    """Tests for IdentifiedEstimand.get_backdoor_variables."""

    def _make_estimand(self, backdoor_variables=None, default_backdoor_id=None, identifier_method=None):
        return IdentifiedEstimand(
            identifier=None,
            treatment_variable="T",
            outcome_variable="Y",
            backdoor_variables=backdoor_variables,
            default_backdoor_id=default_backdoor_id,
            identifier_method=identifier_method,
        )

    def test_returns_empty_when_backdoor_variables_is_none(self):
        """When no backdoor variables are set, get_backdoor_variables returns []."""
        estimand = self._make_estimand(backdoor_variables=None)
        assert estimand.get_backdoor_variables() == []

    def test_returns_empty_when_backdoor_variables_is_empty_dict(self):
        """When backdoor_variables is an empty dict, get_backdoor_variables returns []."""
        estimand = self._make_estimand(backdoor_variables={})
        assert estimand.get_backdoor_variables() == []

    def test_returns_empty_when_default_backdoor_id_is_none(self):
        """When backdoor_variables has entries but default_backdoor_id is None, return [] instead of KeyError.

        Regression test for https://github.com/py-why/dowhy/issues/1335: calling
        get_backdoor_variables() raised KeyError: None when identifier_method was not
        a backdoor method and default_backdoor_id had not been set.
        """
        estimand = self._make_estimand(
            backdoor_variables={"b1": ["W"]},
            default_backdoor_id=None,
            identifier_method="mediation",
        )
        # Must not raise KeyError
        result = estimand.get_backdoor_variables()
        assert result == []

    def test_returns_empty_when_default_backdoor_id_not_in_dict(self):
        """When default_backdoor_id does not exist as a key, return [] instead of KeyError."""
        estimand = self._make_estimand(
            backdoor_variables={"b1": ["W"]},
            default_backdoor_id="nonexistent_key",
        )
        result = estimand.get_backdoor_variables()
        assert result == []

    def test_returns_variables_for_valid_default_backdoor_id(self):
        """When default_backdoor_id is a valid key, return the corresponding variables."""
        estimand = self._make_estimand(
            backdoor_variables={"b1": ["W1", "W2"]},
            default_backdoor_id="b1",
        )
        assert estimand.get_backdoor_variables() == ["W1", "W2"]

    def test_uses_identifier_method_when_backdoor_method(self):
        """When identifier_method starts with 'backdoor', use it as the key."""
        estimand = self._make_estimand(
            backdoor_variables={"backdoor1": ["Z"]},
            default_backdoor_id=None,
            identifier_method="backdoor1",
        )
        assert estimand.get_backdoor_variables() == ["Z"]

    def test_key_parameter_overrides_default(self):
        """Passing an explicit key returns the variables for that key."""
        estimand = self._make_estimand(
            backdoor_variables={"b1": ["W1"], "b2": ["W2"]},
            default_backdoor_id="b1",
        )
        assert estimand.get_backdoor_variables(key="b2") == ["W2"]
