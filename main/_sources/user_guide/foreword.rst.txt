Foreword
=====================


The need for causal inference
----------------------------------

Causal inference is essential for informed decision-making, as it uncovers the true data-generating processes beyond mere associations found in predictive models. It enables us to estimate effects of interventions and counterfactual outcomes, even in the absence of interventional data. Moving beyond correlation-based analysis is vital for generalizing insights and gaining a true understanding of real-world relationships. For instance, for computing systems that actively intervene in societally critical domains such as healthcare, education, and governance, it is important to correctly predict and understand the causal effects of these interventions. Below we describe some common questions in decision-making that require causal inference:

* Will it work?
    * Does a proposed change to a system improve people's outcomes?
* Why did it work?
    * What led to a change in a system's outcome?
* What should we do?
    * What changes to a system are likely to improve outcomes for people?
* What are the overall effects?
    * How does the system interact with human behavior?
    * What is the effect of a system's recommendations on people's activity?

While there are many methods proposed for causal inference, a key challenge is to  compare the assumptions of these different methods and check the robustness of results. To this end, DoWhy is a Python library for causal inference that supports explicit modeling and testing of causal assumptions. DoWhy makes three contributions:

1. Provides a systematic method for modeling causal relationships through graphical representations, ensuring that all underlying assumptions are clearly stated and transparent.
2. Provides a unified interface for many popular causal inference methods, combining the two major frameworks of graphical models and potential outcomes.
3. Provides capabilities to test the validity of assumptions if possible and assesses the robustness of the estimate to violations.

DoWhy's API also provides a unified language for causal inference, combining causal graphical models and potential outcomes frameworks.

