import numpy as np

# Smallest possible value. This is used in various algorithm for numerical stability.
EPS = np.finfo(np.float64).eps

# Constants for falsification of a given graph
FALSIFY_N_VIOLATIONS = 'n_violations'
FALSIFY_N_TESTS = 'n_tests'
FALSIFY_P_VALUE = 'p_value'
FALSIFY_P_VALUES = 'p_values'

FALSIFY_GIVEN_VIOLATIONS = FALSIFY_N_VIOLATIONS + ' g_given'
FALSIFY_PERM_VIOLATIONS = FALSIFY_N_VIOLATIONS + ' permutations'
FALSIFY_LOCAL_VIOLATION_INSIGHT = "local violations"

FALSIFY_METHODS = {
    'validate_lmc': 'LMC',
    'validate_pd': 'Faithfulness',
    'validate_parental_dsep': 'tPa',
    'validate_causal_minimality': 'Causal Minimality'
}

FALSIFY_VIOLATION_COLOR = "red"