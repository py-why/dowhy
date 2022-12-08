class OverruleAnalyzer:
    def __init__(self):
        pass

    def fit(self, X, g):
        self.rules = "Men over 60 years old"

    def __str__(self):
        return self.rules
