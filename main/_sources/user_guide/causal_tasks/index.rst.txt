Performing Causal Tasks
========================
Once we have modeled causal relationships, we are ready to move to the next step: performing causal tasks. DoWhy supports the following tasks.

1. **Effect estimation**: If we change X, how much will it cause Y to change?
2. **Attribution**: Why did an event happen? How to explain an outcome? Which of my variables caused the anomaly?
3. **Counterfactual estimation**: What if X had been changed to a different value than its observed value? What would have been the values of other variables?
4. **Prediction**: Given an input with new values of some input features, what will be the value of Y?

This chapter describes how to perform each of these tasks. 

.. toctree::
   :maxdepth: 2

   estimating_causal_effects/index
   quantify_causal_influence/index
   root_causing_and_explaining/index
   what_if/index
   causal_prediction/index
