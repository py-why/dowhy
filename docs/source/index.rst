DoWhy documentation
===================

.. toctree::
    :maxdepth: 3
    :hidden:
    :glob:

    getting_started/index
    User Guide <user_guide/index>
    Examples <example_notebooks/nb_index>
    cite
    Contributing <contributing>
    dowhy
    code_repo

**Date**: |today| **Version**: |version|


**Related resources**:
`Source Repository <https://github.com/py-why/dowhy>`__ |
`Issues & Ideas <https://github.com/py-why/dowhy/issues>`__ |
`PyWhy Organization <https://www.pywhy.org>`__ |
`DoWhy on PyPI <https://pypi.org/project/dowhy>`__ |

**Join our Community on Discord:** `<https://discord.gg/cSBGb3vsZb>`__

Much like machine learning libraries have done for prediction, DoWhy is a Python library that aims to spark
causal thinking and analysis. DoWhy provides a wide variety of algorithms for effect estimation, causal
structure learning, diagnosis of causal structures, root cause analysis, interventions and counterfactuals.

.. grid:: 1 2 2 2
    :gutter: 4

    .. grid-item-card:: Getting started
        :shadow: md
        :link: getting_started/index
        :link-type: doc

        :octicon:`rocket;2em;sd-text-info`
        ^^^
        New to DoWhy? Our Getting started guide will get you up to speed in minutes. It'll help you install DoWhy and
        write your first lines of code. Once completed, you'll be ready to the run examples and follow along in the
        User Guide.

    .. grid-item-card:: User Guide
        :shadow: md
        :link: user_guide/index
        :link-type: doc

        :octicon:`book;2em;sd-text-info`
        ^^^
        Complete newbie when it comes to causal inference and DoWhy? Then you probably want to read our
        comprehensive User Guide. It guides you through everything you need to know, including the concepts and
        science you need to know when trying to solve non-trivial problems.

    .. grid-item-card:: Examples
        :shadow: md
        :link: example_notebooks/nb_index
        :link-type: doc

        :octicon:`video;2em;sd-text-info`
        ^^^
        If you prefer to learn by example, we recommend to browse the examples. It covers a wide variety of problems
        that you can use to liken to your own problem.

    .. grid-item-card:: API Reference
        :shadow: md
        :link: dowhy
        :link-type: doc

        :octicon:`code;2em;sd-text-info`
        ^^^
        The API reference contains a detailed description of the functions, modules, and objects included in DoWhy.
        It assumes that you have an understanding of the key concepts.


Key differences compared to available causal inference software
----------------------------------------------------------------
DoWhy brings four key differences compared to available software for causal inference:

**Explicit identifying assumptions**
    Assumptions are first-class citizens in DoWhy.

    Each analysis starts by building a causal model. The assumptions can be viewed graphically or in terms
    of conditional independence statements. Further, in the case of GCMs, the data generation process of each node is
    modeled explicitly. Wherever possible, DoWhy can also automatically test stated assumptions using observed data.

**Separation between identification and estimation**
    Identification is the causal problem. Estimation is simply a statistical problem.

    DoWhy respects this boundary and treats them separately. This focuses the causal
    inference effort on identification and frees up estimation using any
    available statistical estimator for a target estimand. In addition, multiple
    estimation methods can be used for a single identified estimand and
    vice-versa. The same goes for modeling causal mechanisms, where any third-party machine learning package can be used for
    modeling the functional relationships.

**Automated validation of assumptions**
    What happens when key identifying assumptions may not be satisfied?

    The most critical, and often skipped, part of causal analysis is checking whether the made assumptions about the
    causal relationships hold. DoWhy makes it easy to automatically run sensitivity and robustness checks on the
    obtained estimate, to falsify the given causal graph, or to evaluate fitted causal mechanisms.

**Default parameters for simple application of complex algorithms**
    Selecting the right set of variables or models is a hard problem.

    DoWhy aims to select appropriate parameters by default while allowing users to fully customize
    each function call and model specification. For instance, DoWhy automatically selects the most appropriate identification
    method or offers functionalities to automatically assign appropriate causal mechanisms.

Finally, DoWhy is easily extensible with a particular focus on supporting other libraries, such as EconML, CausalML,
scikit-learn and more. Algorithms are implemented in a modular way, encouraging users to contribute their own or to
simply plug in different customized models.