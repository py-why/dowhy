Types of graphical causal models
================================

A graphical causal model (GCM) comprises a graphical structure of causal relationships (e.g., X→Y) and causal mechanisms (i.e., :math:`P(Y|X)`)
that describe the (conditional) distribution of each node. In other words, a GCM aims at representing the true data generation process.
For a given set of variables :math:`X_0, ..., X_n`, a GCM models the joint distribution that can be factorized as :math:`P(X_0, ..., X_n) = \prod_i P(X_i|PA_i)`
where :math:`PA_i` are the parents of :math:`X_i`, which could be an empty set in the case of root nodes. The causal
mechanism :math:`P(X_i|PA_i)` may require additional constraints depending on the type of causal questions.

Pearl identifies three levels on the ladder of causality:

**1.** Association: Evaluate and reason using observational data :math:`P(X_0, ..., X_n)`. This level is common for machine
learning prediction problems.

**2.** Intervention: Execute interventions and estimate causal effects. This level is most relevant for causal effect
estimation questions, where we obtain interventional distributions in the form :math:`P(Y|do(X=...))`.

**3.** Counterfactuals: Estimate counterfactual outcomes of observed events. The main difference between interventions and
counterfactuals is that interventions focus on "forward-looking" questions like "What will happen if we change X?",
while counterfactuals examine hypothetical scenarios that deviate from the actual observed event, such as
"What if X had taken a different value, how would our observation have been?".

Estimating counterfactuals in Pearl's framework demands stronger assumptions on causal mechanisms compared to those
needed for estimating interventions. DoWhy offers three different classes with varying degrees of causal mechanism flexibility:

**ProbabilisticCausalModel (PCM):** A PCM provides the greatest flexibility in modeling causal mechanisms, as it only
requires generating samples conditioned on a node's parents. Any model that can handle conditional distributions, such
as a wide array of Bayesian models, can be employed. This class is capable of modeling rungs 1 and 2.

**StructuralCausalModel (SCM):** An SCM limits mechanisms to a deterministic functional causal model (FCM) of parents and
unobserved noise, represented as :math:`X_i = f_i(PA_i, N_i)`, where :math:`N_i` denotes unobserved noise. When fitting the
FCM, the model needs to be capable of accounting for the assumption that :math:`N_i` is unobserved, but can be used as an
input when provided. A SCM generally allows modeling rungs 1 and 2, but additional constraints on the FCM are necessary
to model rung 3.

**InvertibleStructuralCausalModel (ISCM):** An ISCM is an SCM that further requires the underlying FCMs of nodes to be
invertible with respect to noise, meaning the noise can be reconstructed from the observed data. This additional
constraint reduces model flexibility but is required for modeling rung 3. With invertible models, we can address
(sample-specific) counterfactual questions. A typical example of an invertible FCM for continuous variables is an
additive noise model (ANM) of the form :math:`X_i = f_i(PA_i, N_i) = f'_i(PA_i) + N_i`, which can be learned using
standard regression models. The noise then becomes simply the residual.

For causal questions like treatment effect estimation, modeling rung 2 is sufficient. On the other hand, to determine
root causes by asking "Would the target have been an outlier if node X behaved differently?", rung 3 is required. Note
that DoWhy generally uses invertible FCMs, such as ANMs, when models are assigned automatically.