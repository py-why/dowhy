class AdjustmentSet:
    """Class for storing an adjustment set."""

    BACKDOOR = "backdoor"
    GENERAL = "general"

    def __init__(
        self,
        _type,
        variables,
        num_paths_blocked_by_observed_nodes=None,
    ):
        self._type = _type
        self.variables = variables
        self.num_paths_blocked_by_observed_nodes = num_paths_blocked_by_observed_nodes

    def get_type(self):
        """Return the type associated with this adjustment set (backdoor, etc.)"""
        return self._type

    def get_variables(self):
        """Return a list containing the adjustment variables"""
        return self.variables

    def get_num_paths_blocked_by_observed_nodes(self):
        """Return the number of paths blocked by the observed nodes (optional)"""
        return self.num_paths_blocked_by_observed_nodes
