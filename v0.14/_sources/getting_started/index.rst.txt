Getting Started
===============

Installation
^^^^^^^^^^^^

The simplest installation is through `pip <https://pypi.org/project/dowhy/>`__ or conda:

.. tab-set-code::

    .. code-block:: pip

        pip install dowhy

    .. code-block:: conda

        conda install -c conda-forge dowhy

Further installation scenarios and instructions can be found at :doc:`install`.

"Hello causal inference world"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this section, we will show the "Hello world" version of DoWhy. DoWhy is based on a simple unifying language for
causal inference, unifying two powerful frameworks, namely graphical causal models (GCM) and potential outcomes (PO).
It uses graph-based criteria and do-calculus for modeling assumptions and identifying a non-parametric causal effect.

To get you started, we introduce two features out of a large variety of features DoWhy offers.

Effect inference
----------------

For effect estimation, DoWhy switches to methods based primarily on potential outcomes. To do it, DoWhy offers a
simple 4-step recipe consisting of modeling a causal model, identification, estimation, and refutation:

.. code:: python

    from dowhy import CausalModel
    import dowhy.datasets

    # Generate some sample data
    data = dowhy.datasets.linear_dataset(
        beta=10,
        num_common_causes=5,
        num_instruments=2,
        num_samples=10000)

    # Step 1: Create a causal model from the data and given graph.
    model = CausalModel(
        data=data["df"],
        treatment=data["treatment_name"],
        outcome=data["outcome_name"],
        graph=data["gml_graph"])

    # Step 2: Identify causal effect and return target estimands
    identified_estimand = model.identify_effect()

    # Step 3: Estimate the target estimand using a statistical method.
    estimate = model.estimate_effect(identified_estimand,
                                     method_name="backdoor.propensity_score_matching")

    # Step 4: Refute the obtained estimate using multiple robustness checks.
    refute_results = model.refute_estimate(identified_estimand, estimate,
                                           method_name="random_common_cause")

To understand what these four steps mean (and why we need four steps), the best place to learn more is the user
guide's :doc:`../user_guide/causal_tasks/estimating_causal_effects/index` chapter. Alternatively, you can dive into the code and explore
basic features in :doc:`../example_notebooks/dowhy_simple_example`.

For estimation of conditional effects, you can also use methods from `EconML <https://github.com/microsoft/EconML>`_
using the same API, refer to :doc:`../example_notebooks/dowhy-conditional-treatment-effects`.


Graphical causal model-based inference
---------------------------------------

For features like root cause analysis, point-wise counterfactual inference, structural analysis and similar, DoWhy uses
graphical causal models. The language of graphical causal models again offers a variety of causal questions that can
be answered. DoWhy's API to answer these causal questions follows a simple 3-step recipe as follows:

.. code:: python

    import networkx as nx, numpy as np, pandas as pd
    from dowhy import gcm

    # Let's generate some "normal" data we assume we're given from our problem domain:
    X = np.random.normal(loc=0, scale=1, size=1000)
    Y = 2 * X + np.random.normal(loc=0, scale=1, size=1000)
    Z = 3 * Y + np.random.normal(loc=0, scale=1, size=1000)
    data = pd.DataFrame(dict(X=X, Y=Y, Z=Z))

    # Step 1: Model our system:
    causal_model = gcm.StructuralCausalModel(nx.DiGraph([('X', 'Y'), ('Y', 'Z')]))
    gcm.auto.assign_causal_mechanisms(causal_model, data)

    # Step 2: Train our causal model with the data from above:
    gcm.fit(causal_model, data)

    # Step 3: Perform a causal analysis. For instance, root cause analysis, where we observe
    anomalous_sample = pd.DataFrame(dict(X=[0.1], Y=[6.2], Z=[19]))  # Here, Y is the root cause.
    # ... and would like to answer the question:
    # "Which node is the root cause of the anomaly in Z?":
    anomaly_attribution = gcm.attribute_anomalies(causal_model, "Z", anomalous_sample)

If you want to learn more about this and other GCM features, we recommend starting with :doc:`../user_guide/modeling_gcm/index` in
the user guide or check out :doc:`../example_notebooks/gcm_basic_example`.

Further resources
^^^^^^^^^^^^^^^^^

There are further resources available:

- An introductory `tutorial on causal inference <https://github.com/amit-sharma/causal-inference-tutorial/>`_
- A comprehensive
  `tutorial on Causal Inference and Counterfactual Reasoning <https://causalinference.gitlab.io/kdd-tutorial/>`_ at the
  `ACM Knowledge Discovery and Data Mining 2018 conference <http://www.kdd.org/kdd2018/>`_
- A video introduction to the four steps of causal inference and its implications for machine learning from
  Microsoft Research:
  `Foundations of causal inference and its impacts on machine learning <https://note.microsoft.com/MSR-Webinar-DoWhy-Library-Registration-On-Demand.html>`_
- The PDF book `Elements of Causal Inference <https://mitp-content-server.mit.edu/books/content/sectbyfn?collid=books_pres_0&id=11283&fn=11283.pdf>`_
- Draft chapters of an upcoming book: `Causal reasoning: Fundamentals and machine learning applications <https://causalinference.gitlab.io/book/>`_
- A blog post describing one of DoWhy's root cause analysis algorithms via graphical causal models: `New method identifies the root causes of statistical outliers <https://www.amazon.science/blog/new-method-identifies-the-root-causes-of-statistical-outliers>`_


