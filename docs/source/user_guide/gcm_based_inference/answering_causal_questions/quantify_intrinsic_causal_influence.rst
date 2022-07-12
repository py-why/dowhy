Quantifying Intrinsic Causal Influence
======================================

By quantifying intrinsic causal influence, we answer the question:

    How strong is the causal influence of a source node to a target node
    that is not inherited from the parents of the source node?

Naturally, descendants will have a zero intrinsic influence on the target node.

How to use it
^^^^^^^^^^^^^^

To see how the method works, let us generate some data.

>>> import numpy as np, pandas as pd, networkx as nx
>>> from dowhy import gcm
>>> from dowhy.gcm.uncertainty import estimate_variance
>>> np.random.seed(10)  # to reproduce these results

>>> X = np.random.normal(loc=0, scale=1, size=1000)
>>> Y = 2*X + np.random.normal(loc=0, scale=1, size=1000)
>>> Z = 3*Y + np.random.normal(loc=0, scale=1, size=1000)
>>> data = pd.DataFrame(data=dict(X=X, Y=Y, Z=Z))

Next, we will model cause-effect relationships as a structural causal model and fit it to the data.

>>> causal_model = gcm.StructuralCausalModel(nx.DiGraph([('X', 'Y'), ('Y', 'Z')])) # X -> Y -> Z
>>> causal_model.set_causal_mechanism('X', gcm.EmpiricalDistribution())
>>> causal_model.set_causal_mechanism('Y', gcm.AdditiveNoiseModel(gcm.ml.create_linear_regressor()))
>>> causal_model.set_causal_mechanism('Z', gcm.AdditiveNoiseModel(gcm.ml.create_linear_regressor()))
>>> gcm.fit(causal_model, data)

..
    Todo: Use auto module for automatic assignment!

Finally, we can ask for the intrinsic causal influences of ancestors to a node of interest (e.g., :math:`Z`).

>>> contributions = gcm.intrinsic_causal_influence(causal_model, 'Z',
>>>                                               gcm.ml.create_linear_regressor(),
>>>                                               lambda x, _: estimate_variance(x))
>>> contributions
    {'X': 33.34300732332951, 'Y': 9.599478688607254, 'Z': 0.9750701113403872}

**Interpreting the results:** We estimated the intrinsic influence of ancestors of
:math:`Z`, including itself, to its variance. These contributions sum up to the variance of :math:`Z`.
We observe that ~76% of the variance of :math:`Z` comes from :math:`X`.

Understanding the method
^^^^^^^^^^^^^^^^^^^^^^^^^

Consider the following example to get the intuition behind the notion of "intrinsic"
causal influence we seek to measure here.

   A charity event is organised to collect funds to help an orphanage. At the end of the event,
   a donation box is passed around to each participant. Since the donation is voluntary, some may
   not donate for various reasons. For instance, they may not have the cash. In this scenario, a
   participant that simply passes the donation box to the other participant does not contribute
   anything to the collective donation after all. Each person's contribution then is simply the
   amount they donated.

To measure the `intrinsic causal influence <https://arxiv.org/pdf/2007.00714.pdf>`_ of a source
node to a target node, we need a functional causal model. In particular, we assume that the
causal model of each node follows an additive noise model (ANM), i.e. :math:`X_j := f_j
(\textrm{PA}_j) + N_j`, where :math:`\textrm{PA}_j` are the parents of node :math:`X_j` in the causal graph,
and :math:`N_j` is the independent unobserved noise term. To compute the "intrinsic" contribution of ancestors of :math:`X_n` to
some property (e.g. entropy, variance) of the marginal distribution of :math:`X_n`, we first
have to set up our causal graph, and learn the causal model of each node from the dataset.

Consider a causal graph :math:`X \rightarrow Y \rightarrow Z` as in the code example above,
induced by the following ANMs.

.. math::
    X &:= N_X\\
    Y &:= 2 X + N_Y\\
    Z &:= 3 Y + N_Z \;,

where :math:`N_w \sim \mathcal{N}(0, 1)`, for all :math:`w \in \{X, Y, Z\}`, are standard Normal
noise variables.

Suppose that we are interested in the contribution of each variable to the *variance* of the
target :math:`Z`, i.e. :math:`\mathrm{Var}[Z]`. If there were no noise variables, everything can
be contributed to the root node :math:`X` as all other variables would then be its deterministic
function. The intrinsic contribution of each variable to the target quantity
:math:`\mathrm{Var}[Z]` is then really the contribution of corresponding noise term.

To compute "intrinsic" contribution, we also require conditional distributions of :math:`Z` given
subsets of noise variables :math:`N_T`, i.e., :math:`P_{Z \mid
N_T}`, where :math:`T \subseteq \{X, Y, Z\}`. We estimate them using an ANM. To this end,
we have to specify the prediction model from a subset of noise variables to the target. Below, we
quantify the intrinsic causal influence of :math:`X, Y` and :math:`Z` to
:math:`\mathrm{Var}[Z]` using a linear prediction model from noise variables to :math:`Z`.

>>> from dowhy.gcm.uncertainty import estimate_variance
>>> prediction_model_from_noises_to_target = gcm.ml.create_linear_regressor()
>>> node_to_contribution = gcm.intrinsic_causal_influence(causal_model, 'Z',
>>>                                                      prediction_model_from_noises_to_target,
>>>                                                      lambda x, _: estimate_variance(x))

.. note::

  While using variance as uncertainty estimator gives valuable information about the
  contribution of nodes to the squared deviations in the target, one might be rather interested
  in other quantities, such as absolute deviations. This can also be simply computed by replacing
  the uncertainty estimator with a custom function:

  >>> mean_absolute_deviation_estimator = lambda x: np.mean(abs(x))
  >>> node_to_contribution = gcm.intrinsic_causal_influence(causal_model, 'Z',
  >>>                                                      prediction_model_from_noises_to_target,
  >>>                                                      mean_absolute_deviation_estimator)

  If the choice of a prediction model is unclear, the prediction model parameter can also be set
  to "auto".

..
    Todo: Add this once confidence intervals is added!
    Above, we report point estimates of Shapley values from a sample drawn from the estimated joint
    distribution :math:`\hat{P}_{X, Y, Z}`. To quantify the uncertainty of those point estimates, we
    now compute their `bootstrap confidence intervals <https://ocw.mit
    .edu/courses/mathematics/18-05-introduction-to-probability-and-statistics-spring-2014/readings
    /MIT18_05S14_Reading24.pdf>`_ by simply running the above a number of times, and aggregating the
    results.

    >>> from gcm import confidence_intervals, bootstrap_sampling
    >>>
    >>> node_to_mean_contrib, node_to_contrib_conf = confidence_intervals(
    >>>     bootstrap_sampling(gcm.intrinsic_causal_influence, causal_model, 'Z',
    >>>                        prediction_model_from_noises_to_target, lambda x, _: estimate_variance(x)),
    >>>     confidence_level=0.95,
    >>>     num_bootstrap_resamples=200)

    Note that the higher the number of repetitions, the better we are able to approximate the
    sampling distribution of Shapley values.
