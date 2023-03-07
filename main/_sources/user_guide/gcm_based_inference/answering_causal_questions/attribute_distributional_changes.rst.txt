Attributing Distributional Changes
==================================

When attributing distribution changes, we answer the question:

    What mechanism in my system changed between two sets of data?

For example, in a distributed computing system, we want to know why an important system metric changed in a negative way.

How to use it
^^^^^^^^^^^^^^

To see how the method works, let's take the example from above and assume we have a system of three services X, Y, Z,
producing latency numbers. The first dataset ``data_old`` is before the deployment, ``data_new`` is after the
deployment:

>>> import networkx as nx, numpy as np, pandas as pd
>>> from dowhy import gcm
>>> from scipy.stats import halfnorm

>>> X = halfnorm.rvs(size=1000, loc=0.5, scale=0.2)
>>> Y = halfnorm.rvs(size=1000, loc=1.0, scale=0.2)
>>> Z = np.maximum(X, Y) + np.random.normal(loc=0, scale=1, size=1000)
>>> data_old = pd.DataFrame(data=dict(X=X, Y=Y, Z=Z))

>>> X = halfnorm.rvs(size=1000, loc=0.5, scale=0.2)
>>> Y = halfnorm.rvs(size=1000, loc=1.0, scale=0.2)
>>> Z = X + Y + np.random.normal(loc=0, scale=1, size=1000)
>>> data_new = pd.DataFrame(data=dict(X=X, Y=Y, Z=Z))

The change here simulates an accidental conversion of multi-threaded code into sequential one (waiting for X and Y in
parallel vs. waiting for them sequentially).

Next, we'll model cause-effect relationships as a probabilistic causal model:

>>> causal_model = gcm.ProbabilisticCausalModel(nx.DiGraph([('X', 'Z'), ('Y', 'Z')]))  # X -> Z <- Y
>>> causal_model.set_causal_mechanism('X', gcm.EmpiricalDistribution())
>>> causal_model.set_causal_mechanism('Y', gcm.EmpiricalDistribution())
>>> causal_model.set_causal_mechanism('Z', gcm.AdditiveNoiseModel(gcm.ml.create_linear_regressor()))

Finally, we attribute changes in distributions to changes in causal mechanisms:

>>> attributions = gcm.distribution_change(causal_model, data_old, data_new, 'Z')
>>> attributions
{'X': -0.0066425020480165905, 'Y': 0.009816959724738061, 'Z': 0.21957816956354193}

As we can see, :math:`Z` got the highest attribution score here, which matches what we would
expect, given that we changed the mechanism for variable :math:`Z` in our data generation.

As the reader may have noticed, there is no fitting step involved when using this method. The
reason is, that this function will call ``fit`` internally. To be precise, this function will
make two copies of the causal graph and fit one graph to the first dataset and the second graph
to the second dataset.
