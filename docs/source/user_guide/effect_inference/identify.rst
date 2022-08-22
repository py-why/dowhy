Identify a target estimand under the model
----------------------------------------------

Based on the causal graph, DoWhy finds all possible ways of identifying a desired causal effect based on
the graphical model. It uses graph-based criteria and do-calculus to find
potential ways find expressions that can identify the causal effect.

Supported identification criteria
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Back-door criterion
* Front-door criterion
* Instrumental Variables
* Mediation (Direct and indirect effect identification)

Different notebooks illustrate how to use these identification criteria. Check
out the `Simple Backdoor <https://github.com/microsoft/dowhy/blob/main/docs/source/example_notebooks/dowhy_confounder_example.ipynb>`_ notebook for the back-door criterion, and the `Simple IV <https://github.com/microsoft/dowhy/blob/main/docs/source/example_notebooks/dowhy-simple-iv-example.ipynb>`_ notebook for the instrumental variable criterion.

I
