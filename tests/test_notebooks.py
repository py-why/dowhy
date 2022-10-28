import os
import subprocess
import tempfile

import nbformat
import pytest
from pytest import mark

NOTEBOOKS_PATH = "docs/source/example_notebooks/"
notebooks_list = [f.name for f in os.scandir(NOTEBOOKS_PATH) if f.name.endswith(".ipynb")]
advanced_notebooks = [
    # requires stdin input for identify in weighting sampler
    "do_sampler_demo.ipynb",
    # requires Rpy2 for lalonde
    "dowhy_refutation_testing.ipynb",
    "dowhy_lalonde_example.ipynb",
    "lalonde_pandas_api.ipynb",
    # will be removed
    "dowhy_optimize_backdoor_example.ipynb",
    # applied notebook, not necessary to test each time
    "dowhy_ranking_methods.ipynb",
    # needs xgboost too
    "DoWhy-The Causal Story Behind Hotel Booking Cancellations.ipynb",
    #
    # Slow Notebooks
    #
    "tutorial-causalinference-machinelearning-using-dowhy-econml.ipynb",
    "dowhy-conditional-treatment-effects.ipynb",
    "dowhy_refuter_notebook.ipynb",
    "dowhy_twins_example.ipynb",
    "gcm_rca_microservice_architecture.ipynb",
    "gcm_supply_chain_dist_change.ipynb",
    "dowhy_simple_example.ipynb",
    "gcm_401k_analysis.ipynb",
]

ignore_notebooks = [
    # requires Rpy2 for causal discovery
    # daily tests of dowhy_causal_discovery_example.ipynb are failing due to cdt/rpy2 config.
    # comment out, since we are switching causal discovery implementations
    "dowhy_causal_discovery_example.ipynb"
]

# Adding the dowhy root folder to the python path so that jupyter notebooks
# can import dowhy
if "PYTHONPATH" not in os.environ:
    os.environ["PYTHONPATH"] = os.getcwd()
elif os.getcwd() not in os.environ["PYTHONPATH"].split(os.pathsep):
    os.environ["PYTHONPATH"] = os.environ["PYTHONPATH"] + os.pathsep + os.getcwd()


def _notebook_run(filepath):
    """Execute a notebook via nbconvert and collect output.
    :returns (parsed nb object, execution errors)

    Source of this function: http://www.christianmoscardi.com/blog/2016/01/20/jupyter-testing.html
    """
    with tempfile.NamedTemporaryFile(suffix=".ipynb") as fout:
        args = [
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            #          "--ExecutePreprocessor.timeout=600",
            "-y",
            "--no-prompt",
            "--output",
            fout.name,
            filepath,
        ]
        subprocess.check_call(args)

        fout.seek(0)
        nb = nbformat.read(fout, nbformat.current_nbformat)

    errors = [
        output for cell in nb.cells if "outputs" in cell for output in cell["outputs"] if output.output_type == "error"
    ]

    return nb, errors


"""
def test_getstarted_notebook():
    nb, errors = _notebook_run(NOTEBOOKS_PATH+ "dowhy_simple_example.ipynb")
    assert errors == []

def test_confounder_notebook():
    nb, errors = _notebook_run(NOTEBOOKS_PATH+"dowhy_confounder_example.ipynb")
    assert errors = []
"""
parameter_list = []
for nb in notebooks_list:
    if nb in ignore_notebooks:
        continue
    elif nb in advanced_notebooks:
        param = pytest.param(nb, marks=[mark.advanced, mark.notebook], id=nb)
    else:
        param = pytest.param(nb, marks=[mark.notebook], id=nb)
    parameter_list.append(param)


@mark.parametrize("notebook_filename", parameter_list)
def test_notebook(notebook_filename):
    nb, errors = _notebook_run(NOTEBOOKS_PATH + notebook_filename)
    assert errors == []
