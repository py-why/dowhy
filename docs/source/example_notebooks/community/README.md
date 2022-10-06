# Community notebooks

This folder contains notebooks showing use-cases of DoWhy in different applications. 
These notebooks are contributed by users of DoWhy. 

## How to contribute
If you would like to contribute a notebook, raise a Pull Request that adds a new notebook to this folder.
For help with raising a PR, refer to the [instructions for contributing code](https://github.com/py-why/dowhy/blob/main/docs/source/contributing/contributing-code.rst).

Make sure that your notebook conveys the following points:

* What is the problem that the notebook is trying to solve? Ideally, there should be a real-world application/motivation.
* What is the causal question corresponding to the above problem?
* What datasets are being used? How were they collected? A brief description of features in the dataset will be useful.
* Code to show how DoWhy is used to address the causal question.
* Validation of the causal estimate(s). This can be through a mix of refutations and domain knowledge.
* Preferred: A comparison of the results from DoWhy to alternative, non-causal methods.

## Using datasets in your notebook
This folder is intended to store only the notebook code, not the datasets. If you are using an external dataset, it is best to load it dynamically (using its URL) within your code. For example, see this [notebook](https://github.com/py-why/dowhy/blob/main/docs/source/example_notebooks/dowhy_ihdp_data_example.ipynb). If you are bringing in a custom dataset that is not available through a URL, we suggest to host the dataset in a repo on your github account and link to it.

