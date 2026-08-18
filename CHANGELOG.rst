=========
Changelog
=========

All notable changes to DoWhy are documented here.
The format is inspired by `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_.
DoWhy uses `Semantic Versioning <https://semver.org/>`_ starting from v0.8.

.. contents::
   :local:
   :depth: 1

.. _changelog-0.15:

v0.15 (unreleased)
==================

Breaking Changes
----------------

* **Removed AutoGluon integration and BEST quality tier** (`#1688 <https://github.com/py-why/dowhy/pull/1688>`_):
  The ``AssignmentQuality.BEST`` tier (which depended on AutoGluon) has been removed.
  Use ``AssignmentQuality.BETTER`` instead.  AutoGluon is no longer a supported optional dependency.

New Features
------------

* **TabPFN estimator** (`#1392 <https://github.com/py-why/dowhy/pull/1392>`_):
  Added ``TabpfnEstimator`` for causal effect estimation using TabPFN.

* **random_state support in refuters** (`#1556 <https://github.com/py-why/dowhy/pull/1556>`_,
  `#1557 <https://github.com/py-why/dowhy/pull/1557>`_,
  `#1571 <https://github.com/py-why/dowhy/pull/1571>`_):
  ``CausalEstimator``, ``DummyOutcomeRefuter``, and ``AddUnobservedCommonCause``
  now accept ``random_state`` for fully reproducible results.

* **dedicated significance_level parameter** (`#1607 <https://github.com/py-why/dowhy/pull/1607>`_):
  ``estimate_effect()`` now accepts ``significance_level`` directly; the significance
  output is also clearer.

* **CamelCase interpreter lookup** (`#1573 <https://github.com/py-why/dowhy/pull/1573>`_):
  ``CausalEstimate.interpret()`` now accepts interpreter names in CamelCase
  (e.g. ``"ConfounderDistributionInterpreter"``) in addition to the existing snake_case style.

* **Parallelised refutation** (`#1399 <https://github.com/py-why/dowhy/pull/1399>`_):
  Refuters now run simulations in parallel via ``joblib``.

* **random SCM generator** (`#1385 <https://github.com/py-why/dowhy/pull/1385>`_):
  Added a utility to generate random structural causal models for testing and experimentation.

* **time-decay attribution** (`#1388 <https://github.com/py-why/dowhy/pull/1388>`_):
  Added time-decay attribution for GCM and a companion demo notebook.

* **Improved REPL/Jupyter display** (`#1630 <https://github.com/py-why/dowhy/pull/1630>`_):
  ``CausalEstimate``, ``CausalRefutation``, and ``IdentifiedEstimand`` now implement
  ``__repr__`` for nicer output in interactive sessions.

* **Example notebooks** (`#1524 <https://github.com/py-why/dowhy/pull/1524>`_,
  `#1663 <https://github.com/py-why/dowhy/pull/1663>`_):
  New GCM notebooks: auditing CNN predictions for spurious correlations using chest X-ray data;
  causal discovery with a foundation model + GCM root cause analysis.

Improvements
------------

* **Better error messages** (`#1537 <https://github.com/py-why/dowhy/pull/1537>`_,
  `#1594 <https://github.com/py-why/dowhy/pull/1594>`_,
  `#1552 <https://github.com/py-why/dowhy/pull/1552>`_,
  `#1532 <https://github.com/py-why/dowhy/pull/1532>`_):
  ``method_name=None`` now raises a clear ``ValueError``; missing estimand and
  instrumental-variable misuse also surface informative errors.

* **NaN warnings** (`#1599 <https://github.com/py-why/dowhy/pull/1599>`_):
  ``CausalModel`` warns when treatment or outcome columns contain NaN values.

* **Academic citations in docstrings** (`#1598 <https://github.com/py-why/dowhy/pull/1598>`_):
  Key causal estimators now include literature references.

* **Deprecation warning for ``learn_graph()``** (`#1620 <https://github.com/py-why/dowhy/pull/1620>`_):
  ``CausalModel.learn_graph()`` raises ``DeprecationWarning``; users should migrate to
  the dedicated graph-learner API.

* **``CausalEstimator`` pickle/deepcopy safety** (`#1564 <https://github.com/py-why/dowhy/pull/1564>`_):
  Logger is excluded from pickling, preventing ``AttributeError`` on copy.

* **Performance: ``marginal_expectation``** (`#1676 <https://github.com/py-why/dowhy/pull/1676>`_):
  Inner loops in ``marginal_expectation`` vectorised for significant speed-up.

Deprecations
------------

* ``CausalModel.learn_graph()`` is deprecated.  See improvement note above.

Bug Fixes
---------

* **pandas ≥ 3.0 compatibility**
  (`#1609 <https://github.com/py-why/dowhy/pull/1609>`_,
  `#1632 <https://github.com/py-why/dowhy/pull/1632>`_,
  `#1575 <https://github.com/py-why/dowhy/pull/1575>`_,
  `#1586 <https://github.com/py-why/dowhy/pull/1586>`_):
  Fixed crashes due to removed ``include_groups`` parameter, read-only array permute,
  deprecated ``GroupBy.apply`` warning, and categorical confounder handling.

* **scikit-learn ≥ 1.7 compatibility**
  (`#1611 <https://github.com/py-why/dowhy/pull/1611>`_):
  Fixed crash in ``categorical_treatment_model`` due to removed ``multi_class`` parameter.

* **NumPy ≥ 2.4 compatibility**
  (`#1394 <https://github.com/py-why/dowhy/pull/1394>`_,
  `#1426 <https://github.com/py-why/dowhy/pull/1426>`_):
  Fixed binary/categorical treatment generation; replaced removed ``np.float``/``np.int`` aliases.

* **GraphRefuter dtype detection** (`#1595 <https://github.com/py-why/dowhy/pull/1595>`_):
  All integer and bool dtypes (``int8``, ``uint8``, ``int16``, ``bool``, …) are now
  correctly classified as discrete; previously only ``int32``/``int64`` were caught.

* **Conditional-MI entropy computation** (`#1547 <https://github.com/py-why/dowhy/pull/1547>`_):
  Corrected the entropy normalisation in ``conditional_MI`` for discrete variables.

* **PlaceboTreatmentRefuter with IV estimators** (`#1457 <https://github.com/py-why/dowhy/pull/1457>`_):
  Fixed the ``isinstance`` check so instrument substitution works correctly.

* **PlaceboTreatmentRefuter with multiple treatments** (`#1467 <https://github.com/py-why/dowhy/pull/1467>`_):
  No longer raises ``ValueError: Wrong number of items passed N`` when ``num_treatments > 1``.

* **DataSubsetRefuter** (`#1546 <https://github.com/py-why/dowhy/pull/1546>`_,
  `#1560 <https://github.com/py-why/dowhy/pull/1560>`_):
  Fixed index misalignment in subset resampling; each simulation now draws a distinct
  random subset even when a fixed ``random_state`` is used.

* **DistanceMatchingEstimator** (`#1465 <https://github.com/py-why/dowhy/pull/1465>`_,
  `#1634 <https://github.com/py-why/dowhy/pull/1634>`_,
  `#1563 <https://github.com/py-why/dowhy/pull/1563>`_):
  Fixed ``exact_match_cols`` logic; int-encoded binary treatment no longer raises; distance
  metric parameters are passed as ``metric_params`` to ``NearestNeighbors``.

* **AddUnobservedCommonCause** (`#1585 <https://github.com/py-why/dowhy/pull/1585>`_,
  `#1420 <https://github.com/py-why/dowhy/pull/1420>`_):
  Fixed crash when treatment/outcome use ``bool`` dtype; fixed ``IndexError`` in kappa
  auto-inference (``np.std()`` returns a scalar, not an array).

* **TwoStageRegressionEstimator** (`#1520 <https://github.com/py-why/dowhy/pull/1520>`_,
  `#1495 <https://github.com/py-why/dowhy/pull/1495>`_,
  `#1456 <https://github.com/py-why/dowhy/pull/1456>`_):
  ``ValueError`` raised when IV branch has no instruments; pre-instantiated stage models
  now correctly update their ``_target_estimand``; fixed NDE estimand setup.

* **LinearRegressionEstimator p-value** (`#1471 <https://github.com/py-why/dowhy/pull/1471>`_):
  Returns a scalar float p-value for single-treatment models instead of a length-1 array.

* **CausalModel.do()** (`#1576 <https://github.com/py-why/dowhy/pull/1576>`_):
  Fixed ``method_params`` forwarding and ``fit()`` call signature.

* **WeightingSampler** (`#1589 <https://github.com/py-why/dowhy/pull/1589>`_):
  Added descriptive ``ValueError`` for out-of-bounds interventions.

* **NDE/NIE mediator identification** (`#1418 <https://github.com/py-why/dowhy/pull/1418>`_):
  All valid mediators are now identified in natural direct/indirect effect estimation.

* **PropensityBalanceInterpreter** (`#1550 <https://github.com/py-why/dowhy/pull/1550>`_):
  Works with arbitrary covariate and treatment column names.

* **CausalGraph.__init__** (`#1621 <https://github.com/py-why/dowhy/pull/1621>`_):
  Eliminated duplicated DOT/GML parsing logic.

* **GCM bug fixes** (`#1664 <https://github.com/py-why/dowhy/pull/1664>`_):
  Multiple small fixes in the GCM module.

* **Zero-variance refuter p-value** (`#1454 <https://github.com/py-why/dowhy/pull/1454>`_):
  Returns ``p_value=1`` when simulations have zero variance, instead of undefined behaviour.

* **sympy ``init_printing()``** (`#1405 <https://github.com/py-why/dowhy/pull/1405>`_):
  Removed module-level ``sympy.init_printing()`` call that interfered with PyTorch tensor display.

* **OrderedSet iteration** (`#1581 <https://github.com/py-why/dowhy/pull/1581>`_):
  Fixed iteration over falsy elements (``0``, empty strings, etc.).

* **causal_prediction tensor grouping** (`#1371 <https://github.com/py-why/dowhy/pull/1371>`_):
  Fixed tensor grouping bug and optimised MMD calculation.

* **SyntaxWarning: invalid escape sequences** (`#1377 <https://github.com/py-why/dowhy/pull/1377>`_):
  Replaced bare string literals containing backslash sequences with raw strings.

Testing & CI
------------

* **venv caching in CI** (`#1412 <https://github.com/py-why/dowhy/pull/1412>`_):
  CI pipeline speed improved by caching the Poetry virtual environment.

* **EconML in CI** (`#1584 <https://github.com/py-why/dowhy/pull/1584>`_):
  EconML extras are now installed in the initial Poetry install step.

* **AGENTS.md** (`#1422 <https://github.com/py-why/dowhy/pull/1422>`_):
  Added contributor guidelines for AI agents.

* **Additional test coverage** (`#1413 <https://github.com/py-why/dowhy/pull/1413>`_,
  `#1438 <https://github.com/py-why/dowhy/pull/1438>`_,
  `#1555 <https://github.com/py-why/dowhy/pull/1555>`_,
  `#1635 <https://github.com/py-why/dowhy/pull/1635>`_):
  Tests for ``RandomCommonCauseRefuter``, ``causal_prediction``, ``fit_estimator=False``
  caching, and ``choose_variables()``.

.. _changelog-0.14:

v0.14
=====

For changes in earlier versions, refer to the
`GitHub Releases page <https://github.com/py-why/dowhy/releases>`_.
