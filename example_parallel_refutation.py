#!/usr/bin/env python3

"""
Example demonstrating parallel refutation in dowhy.

This shows how to use the new n_jobs parameter to speed up refutation
by running simulations in parallel.
"""

# This is a mock example since we don't have all dependencies installed
# In practice, this would work with real data and full dowhy environment

def example_usage():
    """
    Example of how to use the new parallelization features.
    """
    
    print("Example: Using parallel refutation with dowhy")
    print("=" * 50)
    
    example_code = '''
import dowhy

# Load your data and create a causal model
model = dowhy.CausalModel(
    data=your_data,
    treatment='treatment_variable', 
    outcome='outcome_variable',
    common_causes=['confounder1', 'confounder2']
)

# Identify the causal effect
identified_estimand = model.identify_effect()

# Estimate the causal effect  
causal_estimate = model.estimate_effect(
    identified_estimand,
    method_name="backdoor.linear_regression"
)

print("Original estimate:", causal_estimate.value)

# BEFORE: Sequential refutation (slow)
refute_bootstrap = model.refute_estimate(
    causal_estimate,
    method_name="bootstrap_refuter",
    num_simulations=1000,
    n_jobs=1  # Sequential execution
)

# AFTER: Parallel refutation (fast!)  
refute_bootstrap = model.refute_estimate(
    causal_estimate,
    method_name="bootstrap_refuter", 
    num_simulations=1000,
    n_jobs=-1,  # Use all available CPUs
    verbose=1   # Show progress
)

# Also works for other refuters:

# Parallel placebo treatment refutation
refute_placebo = model.refute_estimate(
    causal_estimate,
    method_name="placebo_treatment_refuter",
    num_simulations=500,
    n_jobs=4,  # Use 4 CPU cores
    verbose=1
)

# Parallel random common cause refutation  
refute_random = model.refute_estimate(
    causal_estimate,
    method_name="random_common_cause",
    num_simulations=500,
    n_jobs=-1,
    verbose=1
)

# Parallel data subset refutation
refute_subset = model.refute_estimate(
    causal_estimate,
    method_name="data_subset_refuter", 
    num_simulations=300,
    n_jobs=4,
    verbose=1
)

# NEW: Parallel dummy outcome refutation
refute_dummy = model.refute_estimate(
    causal_estimate,
    method_name="dummy_outcome_refuter",
    num_simulations=200,
    n_jobs=-1,  # Now supports parallelization!
    verbose=1
)

# NEW: Parallel unobserved common cause simulation
refute_unobserved = model.refute_estimate(
    causal_estimate,
    method_name="add_unobserved_common_cause",
    simulation_method="direct-simulation",
    effect_strength_on_treatment=[0.1, 0.2, 0.3],
    effect_strength_on_outcome=[0.1, 0.2, 0.3], 
    n_jobs=-1,  # Now supports parallelization!
    verbose=1
)

print("All refutations completed with parallel execution!")
'''

    print(example_code)
    
    print("\nKey improvements:")
    print("- ✅ BootstrapRefuter: Already had parallelization")
    print("- ✅ PlaceboTreatmentRefuter: Already had parallelization") 
    print("- ✅ RandomCommonCause: Already had parallelization")
    print("- ✅ DataSubsetRefuter: Already had parallelization")
    print("- 🆕 DummyOutcomeRefuter: NOW has parallelization")
    print("- 🆕 AddUnobservedCommonCause: NOW has parallelization")
    
    print("\nPerformance tips:")
    print("- Use n_jobs=-1 to utilize all CPU cores")
    print("- Use n_jobs=1 for sequential execution (backward compatible)")
    print("- Use n_jobs=N to utilize N specific cores")
    print("- Set verbose=1 to see progress and timing information")
    print("- Larger num_simulations will benefit more from parallelization")

if __name__ == '__main__':
    example_usage()