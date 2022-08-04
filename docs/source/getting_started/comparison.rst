Comparison to other packages
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

DoWhy's effect inference API captures all four steps of causal inference:

1. Model a causal inference problem using assumptions.
2. Identify an expression for the causal effect under these assumptions ("causal estimand").
3. Estimate the expression using statistical methods such as matching or instrumental variables.
4. Finally, verify the validity of the estimate using a variety of robustness checks.

This workflow is captured by four key verbs in DoWhy:

- model
- identify
- estimate
- refute

Using these verbs, DoWhy implements a causal inference engine that can support 
a variety of methods. *model* encodes prior knowledge as a formal causal graph, *identify* uses 
graph-based methods to identify the causal effect, *estimate* uses  
statistical methods for estimating the identified estimand, and finally *refute* 
tries to refute the obtained estimate by testing robustness to assumptions.


Key difference: Causal assumptions as first-class citizens
-----------------------------------------------------------
Due to DoWhy's focus on the full pipeline of causal analysis (not just a single 
step), there are three differences compared to available software for causal inference.

**Explicit identifying assumptions**
    Assumptions are first-class citizens in DoWhy.

    Each analysis starts with a
    building a causal model. The assumptions can be viewed graphically or in terms
    of conditional independence statements. Wherever possible, DoWhy can also
    automatically test for stated assumptions using observed data.

**Separation between identification and estimation**
    Identification is the causal problem. Estimation is simply a statistical problem.

    DoWhy
    respects this boundary and treats them separately. This focuses the causal
    inference effort on identification, and frees up estimation using any
    available statistical estimator for a target estimand. In addition, multiple
    estimation methods can be used for a single identified_estimand and
    vice-versa.

**Automated robustness checks**
    What happens when key identifying assumptions may not be satisfied?

    The most critical, and often skipped, part of causal analysis is checking the
    robustness of an estimate to unverified assumptions. DoWhy makes it easy to
    automatically run sensitivity and robustness checks on the obtained estimate.

Finally, DoWhy is easily extensible, allowing other implementations of the
four verbs to co-exist (e.g., we support implementations of the *estimation* verb from 
EconML and CausalML libraries). The four verbs are mutually independent, so their
implementations can be combined in any way.


