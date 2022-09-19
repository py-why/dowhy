""" Module containing a helper class for creating a Synthetic Control causal model

"""
import logging

import networkx as nx

import dowhy.causal_model as causal_model
from dowhy.causal_graph import CausalGraph

class SyntheticControlGraph(CausalGraph):
    """ Class for creating a Synthetic Control causal graph

    """

    # define a static constant
    SYNTHETIC_CONTROL = "synthetic_control"

    def __init__(self, outcomes, covariates, treatment):
        """ Initialize a SyntheticControlGraph object

        :param outcomes: List of outcome variables
        :param covariates: List of covariates
        :param treatment: treatment variable 
        """

        indexed_treatment = treatment + "_{t}"
        indexed_outcomes = [outcome + "_{t}" for outcome in outcomes]
        indexed_covariates = [covariate + "_{t}" for covariate in covariates]

        g = nx.DiGraph()
        g.add_node(indexed_treatment)
        g.add_nodes_from(indexed_outcomes)
        g.add_nodes_from(indexed_covariates)
        
        # add an edge from treatment to each outcome
        for outcome in indexed_outcomes:
            g.add_edge(treatment, outcome)

        # add an edge from each covariate to each outcome
        for covariate in indexed_covariates:
            for outcome in indexed_outcomes:
                g.add_edge(covariate, outcome)

        super().__init__(graph=g)

        self._treatment = treatment
        self._outcomes = outcomes
        self._covariates = covariates

        self.identification_override = SyntheticControlGraph.SYNTHETIC_CONTROL
        self.logger = logging.getLogger(__name__)
