#!/usr/bin/env python3

"""
Simple test script to verify parallel refuters work correctly.
This doesn't require full dependencies and just tests the basic structure.
"""

import os
import sys

sys.path.insert(0, "/tmp/dowhy-410")


def test_refuter_source_code():
    """Test that refuters have correct source code structure."""
    try:
        # Check dummy_outcome_refuter.py
        with open("/tmp/dowhy-410/dowhy/causal_refuters/dummy_outcome_refuter.py") as f:
            dummy_content = f.read()

        assert "def _refute_once(" in dummy_content, "Missing _refute_once function in dummy_outcome_refuter"
        assert "n_jobs: int = 1" in dummy_content, "Missing n_jobs parameter"
        assert "verbose: int = 0" in dummy_content, "Missing verbose parameter"
        assert "Parallel(n_jobs=n_jobs, verbose=verbose)" in dummy_content, "Missing Parallel call"

        print("✓ dummy_outcome_refuter.py has correct parallelization structure")

        # Check add_unobserved_common_cause.py
        with open("/tmp/dowhy-410/dowhy/causal_refuters/add_unobserved_common_cause.py") as f:
            aucc_content = f.read()

        assert (
            "def _simulate_confounders_effect_once(" in aucc_content
        ), "Missing _simulate_confounders_effect_once function"
        assert "n_jobs: int = 1" in aucc_content, "Missing n_jobs parameter in sensitivity_simulation"
        assert "verbose: int = 0" in aucc_content, "Missing verbose parameter in sensitivity_simulation"
        assert (
            "Parallel(n_jobs=n_jobs, verbose=verbose)" in aucc_content
        ), "Missing Parallel call in sensitivity_simulation"

        print("✓ add_unobserved_common_cause.py has correct parallelization structure")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_joblib_imports():
    """Test that joblib imports are present."""
    try:
        # Check imports exist in the source code
        with open("/tmp/dowhy-410/dowhy/causal_refuters/dummy_outcome_refuter.py") as f:
            dummy_content = f.read()
        assert (
            "from joblib import Parallel, delayed" in dummy_content
        ), "Missing joblib imports in dummy_outcome_refuter"

        with open("/tmp/dowhy-410/dowhy/causal_refuters/add_unobserved_common_cause.py") as f:
            aucc_content = f.read()
        assert (
            "from joblib import Parallel, delayed" in aucc_content
        ), "Missing joblib imports in add_unobserved_common_cause"

        print("✓ joblib imports found in both refuters")
        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def main():
    """Run all tests."""
    print("Testing parallelized refuters...")

    tests = [
        test_refuter_source_code,
        test_joblib_imports,
    ]

    results = []
    for test in tests:
        print(f"\n--- {test.__name__} ---")
        results.append(test())

    print(f"\n=== Results ===")
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    exit(main())
