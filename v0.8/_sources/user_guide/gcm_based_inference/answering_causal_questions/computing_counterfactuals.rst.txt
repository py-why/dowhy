Computing Counterfactuals
==========================

By computing counterfactuals, we answer the question:

    I observed a certain outcome z for a variable Z where variable X was set to a value x. What
    would have happened to the value of Z, had I intervened on X to assign it a different value x'?

As a concrete example, we can imagine the following:

   I'm seeing unhealthy high levels of my `cholesterol LDL
   <https://www.google.com/search?q=cholesterol+ldl>`_ (Z=10). I didn't take any medication
   against it in recent months (X=0). What would have happened to my cholesterol LDL level (Z),
   had I taken a medication dosage of 5g a day (X := 5)?

How to use it
^^^^^^^^^^^^^^

To see how the method works, let's generate some data:

>>> import networkx as nx, numpy as np, pandas as pd
>>> from dowhy import gcm

>>> X = np.random.normal(loc=0, scale=1, size=1000)
>>> Y = 2*X + np.random.normal(loc=0, scale=1, size=1000)
>>> Z = 3*Y + np.random.normal(loc=0, scale=1, size=1000)
>>> training_data = pd.DataFrame(data=dict(X=X, Y=Y, Z=Z))

Next, we'll model cause-effect relationships as an invertible SCM and fit it to the data:

>>> causal_model = gcm.InvertibleStructuralCausalModel(nx.DiGraph([('X', 'Y'), ('Y', 'Z')])) # X -> Y -> Z
>>> causal_model.set_causal_mechanism('X', gcm.EmpiricalDistribution())
>>> causal_model.set_causal_mechanism('Y', gcm.AdditiveNoiseModel(gcm.ml.create_linear_regressor()))
>>> causal_model.set_causal_mechanism('Z', gcm.AdditiveNoiseModel(gcm.ml.create_linear_regressor()))

>>> gcm.fit(causal_model, training_data)

Finally, let's compute the counterfactual when intervening on X:

>>> gcm.counterfactual_samples(
>>>     causal_model,
>>>     {'X': lambda x: 2},
>>>     observed_data=pd.DataFrame(data=dict(X=[1], Y=[2], Z=[3])))
   X         Y         Z
0  2  4.034229  9.073294

As we can see, :math:`X` takes our treatment-/intervention-value of 2, and :math:`Y` and :math:`Z`
take deterministic values, based on our trained causal models and fixed observed data. I.e., based
on the data generation process, if :math:`X = 1`, :math:`Y = 2`, we would expect :math:`Z` to
be 6, but we *observed* :math:`Z = 3`, which means the particular noise value for :math:`Z` in this
particular sample is approximately -2.98. Now, given that we know this hidden noise factor, we can
estimate the counterfactual value of :math:`Z`, had we set :math:`X := 2`, which is approximately
9.07 (as can be seen in the result above).

This shows that the observed data is used to calculate the noise data in the system. We can also
provide these noise values directly, via:

>>> gcm.counterfactual_samples(
>>>     causal_model,
>>>     {'X': lambda x: 2},
>>>     noise_data=pd.DataFrame(data=dict(X=[0], Y=[-0.007913], Z=[-2.97568])))
   X         Y         Z
0  2  4.034229  9.073293

As we see, with :math:`X = 2` and :math:`Y \approx 4.03`, :math:`Z` should be approximately 12. But
we know the hidden noise for this sample, approximately -2.98. So the counterfactual outcome
is again :math:`Z \approx 9.07`.

Understanding the method
^^^^^^^^^^^^^^^^^^^^^^^^

Counterfactuals are very similar to :doc:`simulate_impact_of_interventions`, with an important
difference: when performing interventions, we look into the future, for counterfactuals we look into
an alternative past. To reflect this in the computation, when performing interventions, we generate
all noise using our causal models. For counterfactuals, we use the noise from actual observed data.

To expand on our example above, we assume there are other factors that contribute to cholesterol
levels, e.g. exercising or genetic predisposition. While we *assume* medication helps against high
LDL levels, it's important to take into account all other factors that could also help against it.
We want to prove *what* has helped. Hence, it's important to use the noise from the real data,
not some generated noise from our generative models. Otherwise, I may be able to reduce my
cholesterol LDL level in the counterfactual world, where I take medication (X := 5), but not because
I took the medication, but because the *generated noise* of Z also just happened to be low and so
caused a low value for Z. By taking the *real* noise value of Z (derived from the observed data of
Z), I can prove that it was the medication that helped.
