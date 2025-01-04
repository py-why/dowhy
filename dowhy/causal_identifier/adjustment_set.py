class AdjustmentSet:
    """Class for storing an adjustment set."""

    BACKDOOR = "backdoor"
    # General adjustment sets generalize backdoor sets, but we will differentiate
    # between the two given the ubiquity of the backdoor criterion.
    GENERAL = "general"

    def __init__(
        self,
        adjustment_type,
        adjustment_variables,
        num_paths_blocked_by_observed_nodes=None,
    ):
        self.adjustment_type = adjustment_type
        self.adjustment_variables = adjustment_variables
        self.num_paths_blocked_by_observed_nodes = num_paths_blocked_by_observed_nodes

    def get_adjustment_type(self):
        """Return the technique associated with this adjustment set (backdoor, etc.)"""
        return self.adjustment_type

    def get_adjustment_variables(self):
        """Return a list containing the adjustment variables"""
        return self.adjustment_variables

    def get_num_paths_blocked_by_observed_nodes(self):
        """Return the number of paths blocked by observed nodes (optional)"""
        return self.num_paths_blocked_by_observed_nodes
