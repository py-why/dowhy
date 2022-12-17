Effect inference
=================

.. image:: https://raw.githubusercontent.com/py-why/dowhy/master/docs/images/dowhy-schematic.png

For effect estimation, DoWhy provides a principled four-step interface for causal inference that focuses on
explicitly modeling causal assumptions and validating them as much as possible. The key feature of DoWhy is its
state-of-the-art refutation API that can automatically test causal assumptions for any estimation method, thus making
inference more robust and accessible to non-experts. DoWhy supports estimation of the average causal effect for
backdoor, frontdoor, instrumental variable and other identification methods, and estimation of the conditional effect
(CATE) through an integration with the EconML library.

.. toctree::
   :maxdepth: 1
   :numbered:

   model
   identify
   estimate
   refute
   comparison.rst


