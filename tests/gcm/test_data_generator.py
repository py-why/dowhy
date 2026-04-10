import networkx as nx
import numpy as np
import pandas as pd

from dowhy.gcm.causal_mechanisms import AdditiveNoiseModel
from dowhy.gcm.causal_models import StructuralCausalModel
from dowhy.gcm.data_generator import (
    DataGeneratorConfig,
    _NonAdditiveNoiseFCM,
    _TransformedConditionalModel,
    assign_random_fcms,
    generate_random_dag,
    generate_random_scm,
    generate_samples_from_random_scm,
)
from dowhy.gcm.fitting_sampling import draw_samples
from dowhy.graph import is_root_node


def test_generate_random_dag_node_counts():
    dag = generate_random_dag(3, 5)
    assert dag.number_of_nodes() == 8
    assert nx.is_directed_acyclic_graph(dag)


def test_generate_random_dag_sparse():
    dag = generate_random_dag(5, 10, edge_density=0.0)
    assert dag.number_of_edges() >= 10  # at least 1 parent per child
    assert nx.is_weakly_connected(dag)
    assert nx.is_directed_acyclic_graph(dag)


def test_generate_random_dag_dense():
    dag = generate_random_dag(5, 10, edge_density=1.0)
    max_edges = sum(range(5, 15))  # child i connects to all (5+i) predecessors
    assert dag.number_of_edges() == max_edges


def test_generate_random_scm_returns_structural_causal_model():
    np.random.seed(0)
    scm = generate_random_scm(3, 4)
    assert isinstance(scm, StructuralCausalModel)
    assert scm.graph.number_of_nodes() == 7


def test_generate_random_scm_values_are_bounded():
    np.random.seed(0)
    scm = generate_random_scm(5, 10)

    samples = draw_samples(scm, 2000)
    assert samples.shape == (2000, 15)
    assert samples.min().min() > -20
    assert samples.max().max() < 20


def test_generate_samples_from_random_scm_shape():
    np.random.seed(0)
    samples = generate_samples_from_random_scm(3, 4, 500)
    assert isinstance(samples, pd.DataFrame)
    assert samples.shape == (500, 7)


def test_all_linear_config():
    cfg = DataGeneratorConfig(prob_linear_mechanism=1.0, prob_non_additive_noise=0.0)
    np.random.seed(0)
    scm = generate_random_scm(2, 5, cfg)
    for node in scm.graph.nodes:
        if not is_root_node(scm.graph, node):
            m = scm.causal_mechanism(node)
            base = m._base if isinstance(m, _TransformedConditionalModel) else m
            assert isinstance(base, AdditiveNoiseModel)


def test_all_non_additive_config():
    cfg = DataGeneratorConfig(prob_non_additive_noise=1.0)
    np.random.seed(0)
    scm = generate_random_scm(2, 5, cfg)
    for node in scm.graph.nodes:
        if not is_root_node(scm.graph, node):
            m = scm.causal_mechanism(node)
            base = m._base if isinstance(m, _TransformedConditionalModel) else m
            assert isinstance(base, _NonAdditiveNoiseFCM)


def test_clipped_positive_root_nodes():
    cfg = DataGeneratorConfig(prob_clipped_positive=1.0, prob_clipped_negative=0.0, prob_discretised=0.0)
    np.random.seed(0)
    scm = generate_random_scm(3, 3, cfg)

    samples = draw_samples(scm, 1000)
    for col in samples.columns:
        assert samples[col].min() >= -1e-9


def test_discretised_root_nodes():
    cfg = DataGeneratorConfig(
        prob_discretised=1.0, discrete_num_bins_range=(3, 3), prob_clipped_positive=0.0, prob_clipped_negative=0.0
    )
    np.random.seed(0)
    scm = generate_random_scm(2, 3, cfg)

    samples = draw_samples(scm, 1000)
    for col in samples.columns:
        assert samples[col].nunique() <= 4


def test_assign_random_fcms_on_existing_graph():
    graph = nx.DiGraph([("A", "B"), ("A", "C"), ("B", "C")])
    scm = assign_random_fcms(graph)
    assert isinstance(scm, StructuralCausalModel)
    assert set(scm.graph.nodes) == {"A", "B", "C"}

    samples = draw_samples(scm, 100)
    assert samples.shape == (100, 3)


def test_reproducibility_with_seed():
    np.random.seed(123)
    s1 = generate_samples_from_random_scm(3, 3, 100)
    np.random.seed(123)
    s2 = generate_samples_from_random_scm(3, 3, 100)
    pd.testing.assert_frame_equal(s1, s2)
