Example notebooks
=================

These examples are also available on `GitHub <https://github
.com/py-why/dowhy/tree/main/docs/source/example_notebooks>`_. You can `run them locally <https://docs.jupyter
.org/en/latest/running.html>`_ after cloning `DoWhy <https://github.com/py-why/dowhy>`_ and `installing Jupyter
<https://jupyter.org/install>`_. Or you can run them directly in a web browser using the
`Binder environment <https://mybinder.org/v2/gh/microsoft/dowhy/main?filepath=docs%2Fsource%2F>`_.

Introductory examples
---------------------

.. card-carousel:: 3

    .. card:: :doc:`dowhy_simple_example`

        .. image:: ../_static/effect-estimation-estimand-expression.png
        +++
        | **Level:** Beginner
        | **Task:** Effect estimation

    .. card:: :doc:`dowhy-conditional-treatment-effects`

        +++
        | **Level:** Beginner
        | **Task:** Conditional effect estimation

    .. card:: :doc:`gcm_basic_example`

        .. image:: ../_static/graph-xyz.png
            :width: 50px
            :align: center

        +++
        | **Level:** Beginner
        | **Task:** Intervention via GCM


Real world-inspired examples
----------------------------

.. card-carousel:: 3

    .. card:: :doc:`DoWhy-The Causal Story Behind Hotel Booking Cancellations`

        .. image:: ../_static/hotel-bookings.png
        +++
        | **Level:** Beginner
        | **Task:** Effect estimation

    .. card:: :doc:`dowhy_example_effect_of_memberrewards_program`

        .. image:: ../_static/membership-program-graph.png
        +++
        | **Level:** Beginner
        | **Task:** Effect estimation

    .. card:: :doc:`gcm_rca_microservice_architecture`

        .. image:: ../_static/microservice-architecture.png

        +++
        | **Level:** Beginner
        | **Task:** Root cause analysis, intervention via GCM

    .. card:: :doc:`gcm_401k_analysis`

        +++
        | **Level:** Advanced
        | **Task:** Intervention via GCM

    .. card:: :doc:`gcm_supply_chain_dist_change`

        .. image:: ../_static/supply-chain.png

        +++
        | **Level:** Advanced
        | **Task:** Root cause analysis via GCM

    .. card:: :doc:`gcm_counterfactual_medical_dry_eyes`

        +++
        | **Level:** Advanced
        | **Task:** Counterfactuals via GCM

    .. card:: :doc:`gcm_falsify_dag`

        +++
        | **Level:** Advanced
        | **Task:** Falsifying User-Given DAGs

Examples on benchmarks datasets
-------------------------------

.. card-carousel:: 3

    .. card:: :doc:`dowhy_ihdp_data_example`

        +++
        | **Level:** Advanced
        | **Task:** Effect inference

    .. card:: :doc:`dowhy_lalonde_example`

        +++
        | **Level:** Advanced
        | **Task:** Effect inference

    .. card:: :doc:`dowhy_refutation_testing`

        +++
        | **Level:** Advanced
        | **Task:** Effect inference

    .. card:: :doc:`gcm_401k_analysis`

        +++
        | **Level:** Advanced
        | **Task:** GCM inference

    .. card:: :doc:`lalonde_pandas_api`

        +++
        | **Level:** Advanced
        | **Task:** Do Sampler


Miscellaneous
-------------

.. card-carousel:: 3

    .. card:: :doc:`gcm_draw_samples`

        +++
        | **Level:** Beginner
        | **Task:** GCM inference

    .. card:: :doc:`load_graph_example`

        +++
        | **Level:** Beginner
        | **Task:** Effect inference

    .. card:: :doc:`gcm_draw_samples`

        +++
        | **Level:** Beginner
        | **Task:** GCM inference

    .. card:: :doc:`dowhy-simple-iv-example`

        +++
        | **Level:** Beginner
        | **Task:** Effect inference

    .. card:: :doc:`graph_conditional_independence_refuter`

        +++
        | **Level:** Beginner
        | **Task:** Effect inference

    .. card:: :doc:`dowhy_demo_dummy_outcome_refuter`

        +++
        | **Level:** Beginner
        | **Task:** Effect inference

    .. card:: :doc:`dowhy_multiple_treatments`

        +++
        | **Level:** Advanced
        | **Task:** Effect inference


.. toctree::
   :maxdepth: 1
   :caption: Introductory examples
   :hidden:

   dowhy_simple_example
   gcm_basic_example
   dowhy_confounder_example
   dowhy_estimation_methods
   dowhy_interpreter
   dowhy_causal_discovery_example
   dowhy-conditional-treatment-effects
   dowhy_mediation_analysis
   dowhy_refuter_notebook
   dowhy_causal_api
   do_sampler_demo

.. toctree::
   :maxdepth: 1
   :caption: Real world-inspired examples
   :hidden:

   DoWhy-The Causal Story Behind Hotel Booking Cancellations
   dowhy_example_effect_of_memberrewards_program
   gcm_rca_microservice_architecture
   gcm_401k_analysis
   gcm_supply_chain_dist_change
   gcm_counterfactual_medical_dry_eyes
   gcm_falsify_dag

.. toctree::
   :maxdepth: 1
   :caption: Examples on benchmarks datasets
   :hidden:

   dowhy_ihdp_data_example
   dowhy_lalonde_example
   dowhy_refutation_testing
   gcm_401k_analysis
   lalonde_pandas_api

.. toctree::
   :maxdepth: 1
   :caption: Miscellaneous
   :hidden:

   load_graph_example
   gcm_draw_samples
   dowhy-simple-iv-example
   graph_conditional_independence_refuter
   dowhy_demo_dummy_outcome_refuter
   dowhy_multiple_treatments
   identifying_effects_using_id_algorithm.ipynb


