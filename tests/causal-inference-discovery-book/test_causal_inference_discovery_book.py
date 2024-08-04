"""
Note: The tests below are taken from the book 'Causal Inference and Discovery in Python' (https://www.packtpub.com/product/causal-inference-and-discovery-in-python/9781804612989)
by Aleksander Molak (https://github.com/AlxndrMlk). Other than the assert statements, all of the code below is taken from the book.
"""

import numpy as np
import pandas as pd
from pytest import mark
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LassoCV, LinearRegression, LogisticRegression
from tqdm import tqdm

from dowhy import CausalModel


class GPSMemorySCM:
    def __init__(self, random_seed=None):
        self.random_seed = random_seed
        self.u_x = stats.truncnorm(0, np.infty, scale=5)
        self.u_y = stats.norm(scale=2)
        self.u_z = stats.norm(scale=2)
        self.u = stats.truncnorm(0, np.infty, scale=4)

    def sample(self, sample_size=100, treatment_value=None):
        """Samples from the SCM"""
        if self.random_seed:
            np.random.seed(self.random_seed)

        u_x = self.u_x.rvs(sample_size)
        u_y = self.u_y.rvs(sample_size)
        u_z = self.u_z.rvs(sample_size)
        u = self.u.rvs(sample_size)

        if treatment_value:
            gps = np.array([treatment_value] * sample_size)
        else:
            gps = u_x + 0.7 * u

        hippocampus = -0.6 * gps + 0.25 * u_z
        memory = 0.7 * hippocampus + 0.25 * u

        return gps, hippocampus, memory

    def intervene(self, treatment_value, sample_size=100):
        """Intervenes on the SCM"""
        return self.sample(treatment_value=treatment_value, sample_size=sample_size)


@mark.usefixtures("fixed_seed")
class TestCausalInferenceDiscoveryBook(object):
    def test_dowhy_chapter_8(self):
        """
        This test was taken from Chapter 8 of the book, 'Causal Inference and Discovery in Python'
        """

        # Construct the graph (the graph is constant for all iterations)
        nodes = ["S", "Q", "X", "Y", "P"]
        edges = ["SQ", "SY", "QX", "QY", "XP", "YP", "XY"]

        # Generate the GML graph
        gml_string = "graph [directed 1\n"

        for node in nodes:
            gml_string += f'\tnode [id "{node}" label "{node}"]\n'

        for edge in edges:
            gml_string += f'\tedge [source "{edge[0]}" target "{edge[1]}"]\n'

        gml_string += "]"

        # Define the true effect
        TRUE_EFFECT = 0.7

        # Define experiment params
        sample_sizes = [30, 100, 1000, 10000]
        noise_coefs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        n_samples = 20

        # Record the results
        results = []

        # Run the experiment
        for sample_size in tqdm(sample_sizes):
            for noise_coef in noise_coefs:
                for i in range(n_samples):
                    # Generate the data
                    S = np.random.random(sample_size)
                    Q = 0.2 * S + noise_coef * np.random.random(sample_size)
                    X = 0.14 * Q + noise_coef * np.random.random(sample_size)
                    Y = TRUE_EFFECT * X + 0.11 * Q + 0.32 * S + noise_coef * np.random.random(sample_size)
                    P = 0.43 * X + 0.21 * Y + noise_coef * np.random.random(sample_size)

                    # Encode as a pandas df
                    df = pd.DataFrame(np.vstack([S, Q, X, Y, P]).T, columns=["S", "Q", "X", "Y", "P"])

                    # Instantiate the CausalModel
                    model = CausalModel(data=df, treatment="X", outcome="Y", graph=gml_string)

                    # Get the estimand
                    estimand = model.identify_effect()

                    # Get estimate (DML)
                    estimate_dml = model.estimate_effect(
                        identified_estimand=estimand,
                        method_name="backdoor.econml.dml.DML",
                        method_params={
                            "init_params": {
                                "model_y": GradientBoostingRegressor(),
                                "model_t": GradientBoostingRegressor(),
                                "model_final": LassoCV(fit_intercept=False),
                            },
                            "fit_params": {},
                        },
                    )

                    # Get estimate (Linear Regression)
                    estimate_lr = model.estimate_effect(
                        identified_estimand=estimand, method_name="backdoor.linear_regression"
                    )

                    results.append(
                        {
                            "sample_size": sample_size,
                            "noise_coef": noise_coef,
                            "estimate_dml": estimate_dml.value,
                            "estimate_lr": estimate_lr.value,
                            "error_dml": estimate_dml.value - TRUE_EFFECT,
                            "error_lr": estimate_lr.value - TRUE_EFFECT,
                        }
                    )

    def test_dowhy_chapter_7(self):
        """
        This test was taken from Chapter 7 of the book, 'Causal Inference and Discovery in Python'
        """

        # Instantiate the SCM
        scm = GPSMemorySCM()
        # Generate observational data
        gps_obs, hippocampus_obs, memory_obs = scm.sample(1000)
        # Encode as a pandas df
        df = pd.DataFrame(np.vstack([gps_obs, hippocampus_obs, memory_obs]).T, columns=["X", "Z", "Y"])

        # Create the graph describing the causal structure
        gml_graph = """
        graph [
            directed 1
            
            node [
                id "X" 
                label "X"
            ]    
            node [
                id "Z"
                label "Z"
            ]
            node [
                id "Y"
                label "Y"
            ]
            node [
                id "U"
                label "U"
            ]
            
            edge [
                source "X"
                target "Z"
            ]
            edge [
                source "Z"
                target "Y"
            ]
            edge [
                source "U"
                target "X"
            ]
            edge [
                source "U"
                target "Y"
            ]
        ]
        """

        # With graph
        model = CausalModel(data=df, treatment="X", outcome="Y", graph=gml_graph)

        # model.view_model()

        estimand = model.identify_effect()

        estimate = model.estimate_effect(identified_estimand=estimand, method_name="frontdoor.two_stage_regression")

        refute_subset = model.refute_estimate(
            estimand=estimand, estimate=estimate, method_name="data_subset_refuter", subset_fraction=0.4
        )

        assert refute_subset is not None
