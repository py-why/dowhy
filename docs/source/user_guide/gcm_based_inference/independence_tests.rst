Independence Tests
===================

Assuming we have the following data:

>>> import numpy as np, pandas as pd
>>>
>>> X = np.random.normal(loc=0, scale=1, size=1000)
>>> Y = 2 * X + np.random.normal(loc=0, scale=1, size=1000)
>>> Z = 3 * Y + np.random.normal(loc=0, scale=1, size=1000)
>>> data = pd.DataFrame(data=dict(X=X, Y=Y, Z=Z))

To test whether :math:`X` is conditionally independent of :math:`Z` given :math:`Y` using the
`kernel dependence measure
<https://papers.nips.cc/paper/3201-a-kernel-statistical-test-of-independence.pdf>`_, all you need to
do is:

>>> import dowhy.gcm as gcm
>>>
>>> # Null hypothesis: x is independent of y given z
>>> p_value = gcm.independence_test(X, Z, conditioned_on=Y)
>>> p_value
0.48386151342564865

If we define a threshold of 0.05 (as is often done as a good default), and the p-value is clearly
above this, it says :math:`X` and :math:`Z` are indeed independent when we condition on :math:`Y`.
This is what we would expect, given that we generated the data using the causal graph :math:`X
\rightarrow Y \rightarrow Z`, where Z is conditionally independent of :math:`X` given :math:`Y`.

To test whether :math:`X` is independent of :math:`Z` (*without* conditioning on :math:`Y`), we can
use the same function without the third argument.

>>> # Null hypothesis: x is independent of y
>>> p_value = gcm.independence_test(X, Z)
>>> p_value
0.0

Again, we can define a threshold of 0.05, but this time the p-value is clearly below this threshold.
This says :math:`X` and :math:`Z` *are* dependent on each other. Again, this is what we would
expect, since :math:`Z` is dependent on :math:`Y` and :math:`Y` is dependent on :math:`X`, but we
don't condition on :math:`Y`.
