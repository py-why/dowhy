DoWhy documentation
===================

.. toctree::
    :maxdepth: 3
    :hidden:
    :glob:

    getting_started/index
    User Guide <user_guide/index>
    Examples <example_notebooks/nb_index>
    dowhy
    Contributing <contributing>
    code_repo


.. image:: https://raw.githubusercontent.com/py-why/dowhy/master/docs/images/dowhy-schematic.png

Much like machine learning libraries have done for prediction, **"DoWhy" is a Python library that aims to spark
causal thinking and analysis**. DoWhy provides a principled four-step interface for causal inference that focuses on
explicitly modeling causal assumptions and validating them as much as possible. The key feature of DoWhy is its
state-of-the-art refutation API that can automatically test causal assumptions for any estimation method, thus making
inference more robust and accessible to non-experts. DoWhy supports estimation of the average causal effect for
backdoor, frontdoor, instrumental variable and other identification methods, and estimation of the conditional effect
(CATE) through an integration with the EconML library.

Getting started
---------------

New to DoWhy? Our :doc:`getting_started/index` guide will get you up to speed in minutes. Once completed, you'll be
ready to check out our :doc:`example_notebooks/nb_index`, :doc:`user_guide/index`, and other sections.

User Guide
----------

Complete newbie when it comes to causal inference and DoWhy? Then you probably want to read our
comprehensive :doc:`user_guide/index`. It guides you through everything you need to know, including the concepts and
science you need to know when trying to solve non-trivial problems.

Examples
--------

If you prefer to learn by example, we recommend our :doc:`example_notebooks/nb_index`.

API Reference
-------------

The :doc:`dowhy` guide contains a detailed description of the functions, modules, and objects included in DoWhy.
The reference describes how the methods work and which parameters can be used. It assumes that you have an
understanding of the key concepts.

Contributing
------------

Want to add to the codebase or documentation? Check out our :doc:`contributing` guide.

.. include:: cite.rst

