"""Module for generating some sample datasets."""

import math
import string

import networkx as nx
import numpy as np
import pandas as pd
import scipy.stats as ss
from numpy.random import choice
from scipy.stats import bernoulli, halfnorm, poisson, uniform
from sklearn.neural_network import MLPRegressor

from dowhy.utils.graph_operations import add_edge, del_edge, get_random_node_pair, get_simple_ordered_tree

DISCRETE = "discrete"
BINARY = "binary"
CONTINUOUS = "continuous"


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def convert_to_binary(x, stochastic=True):
    p = sigmoid(x)
    if stochastic:
        return choice([0, 1], 1, p=[1 - p, p])
    else:
        return int(p > 0.5)


def stochastically_convert_to_three_level_categorical(x):
    p = sigmoid(x)
    return choice([0, 1, 2], 1, p=[0.8 * (1 - p), 0.8 * p, 0.2])


def convert_to_categorical(arr, num_vars, num_discrete_vars, quantiles=[0.25, 0.5, 0.75], one_hot_encode=False):
    arr_with_dummy = arr.copy()
    # Below loop assumes that the last indices of W are alwawys converted to discrete
    for arr_index in range(num_vars - num_discrete_vars, num_vars):
        # one-hot encode discrete W
        arr_bins = np.quantile(arr[:, arr_index], q=quantiles)
        arr_categorical = np.digitize(arr[:, arr_index], bins=arr_bins)
        if one_hot_encode:
            dummy_vecs = np.eye(len(quantiles) + 1)[arr_categorical]
            arr_with_dummy = np.concatenate((arr_with_dummy, dummy_vecs), axis=1)
        else:
            arr_with_dummy = np.concatenate((arr_with_dummy, arr_categorical[:, np.newaxis]), axis=1)
    # Now deleting the old continuous value
    for arr_index in range(num_vars - 1, num_vars - num_discrete_vars - 1, -1):
        arr_with_dummy = np.delete(arr_with_dummy, arr_index, axis=1)
    return arr_with_dummy


def construct_col_names(name, num_vars, num_discrete_vars, num_discrete_levels, one_hot_encode):
    colnames = [(name + str(i)) for i in range(0, num_vars - num_discrete_vars)]
    if one_hot_encode:
        discrete_colnames = [
            name + str(i) + "_" + str(j)
            for i in range(num_vars - num_discrete_vars, num_vars)
            for j in range(0, num_discrete_levels)
        ]
        colnames = colnames + discrete_colnames
    else:
        colnames = colnames + [(name + str(i)) for i in range(num_vars - num_discrete_vars, num_vars)]

    return colnames


def linear_dataset(
    beta,
    num_common_causes,
    num_samples,
    num_instruments=0,
    num_effect_modifiers=0,
    num_treatments=None,
    num_frontdoor_variables=0,
    treatment_is_binary=True,
    treatment_is_category=False,
    outcome_is_binary=False,
    stochastic_discretization=True,
    num_discrete_common_causes=0,
    num_discrete_instruments=0,
    num_discrete_effect_modifiers=0,
    stddev_treatment_noise=1,
    stddev_outcome_noise=0.01,
    one_hot_encode=False,
):
    """
    Generate a synthetic dataset with a known effect size.

    This function generates a pandas dataFrame with num_samples records. The variables follow a naming convention where the first letter indicates its role in the causality graph and then a sequence number.

    :param beta: coefficient of the treatment(s) ('v?') in the generating equation of the outcome ('y').
    :type beta: int or list/ndarray of length num_treatments of type int
    :param num_common_causes: Number of variables affecting both the treatment and the outcome [w -> v; w -> y]
    :type num_common_causes: int
    :param num_samples: Number of records to generate
    :type num_samples: int
    :param num_instruments: Number of instrumental variables  [z -> v], defaults to 0
    :type num_instruments: int
    :param num_effect_modifiers: Number of effect modifiers, variables affecting only the outcome [x -> y], defaults to 0
    :type num_effect_modifiers: int
    :param num_treatments: Number of treatment variables [v]. By default inferred from the beta argument. When provided, beta is recycled to match num_treatments.
    :type num_treatments : Union[None, int]
    :param num_frontdoor_variables : Number of frontdoor mediating variables [v -> FD -> y], defaults to  0
    :type num_frontdoor_variables: int
    :param treatment_is_binary: Cannot be True if treatment_is_category is True, defaults to True
    :type treatment_is_binary: bool
    :param treatment_is_category: Cannot be True if treatment_is_binary is True, defaults to False
    :type treatment_is_category: bool
    :param outcome_is_binary: defaults to False,
    :type outcome_is_binary: bool
    :param stochastic_discretization: if False, quartiles are used when discretised variables are specified. They can be hot encoded, defaults True
    :type stochastic_discretization: bool
    :param num_discrete_common_causes: Number of discrete common causes of the total num_common_causes, defaults to 0
    :type num_discrete_common_causes: int
    :param num_discrete_instruments: Number of discrete instrumental variables of the total num_instruments, defaults to 0
    :type num_discrete_instruments  : int
    :param num_discrete_effect_modifiers : Number of discrete effect modifiers of the total effect_modifiers, defaults to 0
    :type num_discrete_effect_modifiers: int
    :param stddev_treatment_noise : defaults to 1
    :type stddev_treatment_noise : float
    :param stddev_outcome_noise: defaults to 0.01
    :type stddev_outcome_noise: float
    :param one_hot_encode: defaults to False
    :type one_hot_encode: bool

    :returns: Dictionary with pandas dataFrame and few other metadata variables.
                        "df": pd.dataFrame
                        with num_samples records. The variables follow a naming convention were the first letter indicates its role in the causality graph and then a sequence number.

                    v variables - are the treatments. They can be binary or continuous. In the case of continuous abs(*beta*) defines thier magnitude;

                    y - is the outcome variable. The generating equation is,
                     y = normal(0, stddev_outcome_noise) + t @ beta [where @ is a numpy matrix multiplication allowing for beta be a vector]

                    W variables - commonly cause both the treatment and the outcome and are iid. if continuous, they are Norm(mu = Unif(-1,1), sigma = 1)

                    Z variables - Instrument variables. Each one affects all treatments. i.e. if there is one instrument and two treatments then z0->v0, z0->v1

                    X variables - effect modifiers. If continuous, they are Norm(mu = Unif(-1,1), sigma = 1)

                    FD variables - Front door variables, v0->FD0->y

            "treatment_name": str/list(str)
            "outcome_name": str
            "common_causes_names": str/list(str)
            "instrument_names": str/list(str)
            "effect_modifier_names": str/list(str)
            "frontdoor_variables_names": str/list(str)
            "dot_graph": dot_graph,
            "gml_graph": gml_graph,
            "ate": float, the true ate in the dataset
    :rtype: dict

    Examples
    ********
    .. code-block:: python
            import networkx as nx
            import matplotlib.pyplot as plt
            import pandas as pd
            import numpy as np
            import dowhy.datasets

            def plot_gml(gml_graph):
                    G = nx.parse_gml(gml_graph)
                    pos=nx.spring_layout(G)
                    nx.draw_networkx(G, pos, with_labels=True, node_size=1000, node_color="darkorange")
                    return(plt.show())

            def describe_synthetic_data(synthetic_data):
                    if (synthetic_data['gml_graph'] != None) :
                    plot_gml(synthetic_data["gml_graph"])
                    synthetic_data_df=synthetic_data["df"]
                    print('------- Variables --------')
                    print('Treatment vars:'      , synthetic_data['treatment_name'])
                    print('Outcome vars:'        , synthetic_data['outcome_name'])
                    print('Common causes vars:'  , synthetic_data['common_causes_names'])
                    print('Instrument vars:'     , synthetic_data['instrument_names'])
                    print('Effect Modifier vars:', synthetic_data['effect_modifier_names'])
                    print('Frontdoor vars:'      , synthetic_data['frontdoor_variables_names'])
                    print('Treatment vars:', synthetic_data['outcome_name'])
                    print('-------- Corr -------')
                    print(synthetic_data_df.corr())
                    print('------- Head --------')
                    return(synthetic_data_df)

            # create a dataset with 10 observations one binary treatment and a continuous outcome affected by one common cause
            synthetic_data = dowhy.datasets.linear_dataset(beta = 100,
                    num_common_causes = 1,
                    num_samples =10
                    )
            describe_synthetic_data(synthetic_data).head()

            # Two continuous treatments, no common cause, an instrumental variable and two effect modifiers - linearly added appropriately
            synthetic_data = dowhy.datasets.linear_dataset(
                    beta                          = 100,
                    num_common_causes             = 0,
                    num_samples                   = 20,
                    num_instruments               = 1,
                    num_effect_modifiers          = 2,
                    num_treatments                = 2,
                    num_frontdoor_variables       = 0,
                    treatment_is_binary           = False,
                    treatment_is_category         = False,
                    outcome_is_binary             = False,
                    stochastic_discretization     = True,
                    num_discrete_common_causes    = 0,
                    num_discrete_instruments      = 0,
                    num_discrete_effect_modifiers = 0,
                    stddev_treatment_noise        = 1,
                    stddev_outcome_noise          = 0.01,
                    one_hot_encode                = False
                    )
            describe_synthetic_data(synthetic_data).head()

            # One Hot Encoding
            synthetic_data = dowhy.datasets.linear_dataset(
                    beta                          = 100,
                    num_common_causes             =   2,
                    num_samples                   =  20,
                    num_instruments               =   1,
                    num_effect_modifiers          =   1,
                    num_treatments                =   1,
                    num_frontdoor_variables       =   1,
                    treatment_is_binary           = False,
                    treatment_is_category         = False,
                    outcome_is_binary             = False,
                    stochastic_discretization     = True,
                    num_discrete_common_causes    = 1, #of the total num_common_causes
                    num_discrete_instruments      = 1,
                    num_discrete_effect_modifiers = 1,
                    stddev_treatment_noise        = 1,
                    stddev_outcome_noise          = 0.01,
                    one_hot_encode                = True
                    )
                    describe_synthetic_data(synthetic_data).head()
    """
    assert not (treatment_is_binary and treatment_is_category)
    W, X, Z, FD, c1, c2, ce, cz, cfd1, cfd2 = [None] * 10
    W_with_dummy, X_with_categorical = (None, None)
    beta = np.array(beta, dtype=np.float64, ndmin=1)
    if num_treatments is None:
        num_treatments = beta.size
    else:
        beta = np.resize(beta, num_treatments)
    num_cont_common_causes = num_common_causes - num_discrete_common_causes
    num_cont_effect_modifiers = num_effect_modifiers - num_discrete_effect_modifiers
    if num_common_causes > 0:
        range_c1 = 0.5 + np.max(np.absolute(beta)) * 0.5
        range_c2 = 0.5 + np.max(np.absolute(beta)) * 0.5
        means = np.random.uniform(-1, 1, num_common_causes)
        cov_mat = np.diag(np.ones(num_common_causes))
        W = np.random.multivariate_normal(means, cov_mat, num_samples)
        W_with_dummy = convert_to_categorical(
            W, num_common_causes, num_discrete_common_causes, quantiles=[0.25, 0.5, 0.75], one_hot_encode=one_hot_encode
        )
        c1 = np.random.uniform(0, range_c1, (W_with_dummy.shape[1], num_treatments))
        c2 = np.random.uniform(0, range_c2, W_with_dummy.shape[1])

    if num_instruments > 0:
        range_cz = 1 + np.max(np.absolute(beta))
        p = np.random.uniform(0, 1, num_instruments)
        Z = np.zeros((num_samples, num_instruments))
        for i in range(num_instruments):
            if (i % 2) == 0:
                Z[:, i] = np.random.binomial(n=1, p=p[i], size=num_samples)
            else:
                Z[:, i] = np.random.uniform(0, 1, size=num_samples)
        # TODO Ensure that we do not generate weak instruments
        cz = np.random.uniform(
            range_cz - (range_cz * 0.05), range_cz + (range_cz * 0.05), (num_instruments, num_treatments)
        )
    if num_effect_modifiers > 0:
        range_ce = 0.5 + np.max(np.absolute(beta)) * 0.5
        means = np.random.uniform(-1, 1, num_effect_modifiers)
        cov_mat = np.diag(np.ones(num_effect_modifiers))
        X = np.random.multivariate_normal(means, cov_mat, num_samples)
        X_with_categorical = convert_to_categorical(
            X,
            num_effect_modifiers,
            num_discrete_effect_modifiers,
            quantiles=[0.25, 0.5, 0.75],
            one_hot_encode=one_hot_encode,
        )
        ce = np.random.uniform(0, range_ce, X_with_categorical.shape[1])
    # TODO - test all our methods with random noise added to covariates (instead of the stochastic treatment assignment)

    t = np.random.normal(0, stddev_treatment_noise, (num_samples, num_treatments))
    if num_common_causes > 0:
        t += W_with_dummy @ c1  # + np.random.normal(0, 0.01)
    if num_instruments > 0:
        t += Z @ cz
    # Converting treatment to binary if required
    if treatment_is_binary:
        t = np.vectorize(convert_to_binary)(t)
    elif treatment_is_category:
        t = np.vectorize(stochastically_convert_to_three_level_categorical)(t)

    # Generating frontdoor variables if asked for
    if num_frontdoor_variables > 0:
        range_cfd1 = np.max(np.absolute(beta)) * 0.5
        range_cfd2 = np.max(np.absolute(beta)) * 0.5
        cfd1 = np.random.uniform(0, range_cfd1, (num_treatments, num_frontdoor_variables))
        cfd2 = np.random.uniform(0, range_cfd2, num_frontdoor_variables)
        FD_noise = np.random.normal(0, 1, (num_samples, num_frontdoor_variables))
        FD = FD_noise
        FD += t @ cfd1
        if num_common_causes > 0:
            range_c1_frontdoor = range_c1 / 10.0
            c1_frontdoor = np.random.uniform(0, range_c1_frontdoor, (W_with_dummy.shape[1], num_frontdoor_variables))
            FD += W_with_dummy @ c1_frontdoor

    def _compute_y(t, W, X, FD, beta, c2, ce, cfd2, stddev_outcome_noise):
        y = np.random.normal(0, stddev_outcome_noise, num_samples)
        if num_frontdoor_variables > 0:
            y += FD @ cfd2
        else:
            # NOTE: We are assuming a linear relationship *even when t is categorical* and integer coded.
            # For categorical t, this example dataset has the effect size for category 2 being exactly
            # double the effect for category 1
            # This could be changed at this stage by one-hot encoding t and using a custom beta that
            # sets a different effect for each category {0, 1, 2}
            y += t @ beta
        if num_common_causes > 0:
            y += W @ c2
        if num_effect_modifiers > 0:
            y += (X @ ce) * np.prod(t, axis=1)
        if outcome_is_binary:
            y = np.vectorize(convert_to_binary)(y, stochastic_discretization)
        return y

    y = _compute_y(t, W_with_dummy, X_with_categorical, FD, beta, c2, ce, cfd2, stddev_outcome_noise)

    data = np.column_stack((t, y))
    if num_common_causes > 0:
        data = np.column_stack((W_with_dummy, data))
    if num_instruments > 0:
        data = np.column_stack((Z, data))
    if num_effect_modifiers > 0:
        data = np.column_stack((X_with_categorical, data))
    if num_frontdoor_variables > 0:
        data = np.column_stack((FD, data))

    # Computing ATE
    FD_T1, FD_T0 = None, None
    T1 = np.ones((num_samples, num_treatments))
    T0 = np.zeros((num_samples, num_treatments))
    if num_frontdoor_variables > 0:
        FD_T1 = FD_noise + (T1 @ cfd1)
        FD_T0 = FD_noise + (T0 @ cfd1)
    ate = np.mean(
        _compute_y(T1, W_with_dummy, X_with_categorical, FD_T1, beta, c2, ce, cfd2, stddev_outcome_noise)
        - _compute_y(T0, W_with_dummy, X_with_categorical, FD_T0, beta, c2, ce, cfd2, stddev_outcome_noise)
    )

    treatments = [("v" + str(i)) for i in range(0, num_treatments)]
    outcome = "y"
    # constructing column names for one-hot encoded discrete features
    common_causes = construct_col_names(
        "W", num_common_causes, num_discrete_common_causes, num_discrete_levels=4, one_hot_encode=one_hot_encode
    )
    instruments = [("Z" + str(i)) for i in range(0, num_instruments)]
    frontdoor_variables = [("FD" + str(i)) for i in range(0, num_frontdoor_variables)]
    effect_modifiers = construct_col_names(
        "X", num_effect_modifiers, num_discrete_effect_modifiers, num_discrete_levels=4, one_hot_encode=one_hot_encode
    )
    other_variables = None
    col_names = frontdoor_variables + effect_modifiers + instruments + common_causes + treatments + [outcome]
    data = pd.DataFrame(data, columns=col_names)
    # Specifying the correct dtypes
    if treatment_is_binary:
        data = data.astype({tname: "bool" for tname in treatments}, copy=False)
    elif treatment_is_category:
        data = data.astype({tname: "category" for tname in treatments}, copy=False)
    if outcome_is_binary:
        data = data.astype({outcome: "bool"}, copy=False)
    if num_discrete_common_causes > 0 and not one_hot_encode:
        data = data.astype({wname: "int64" for wname in common_causes[num_cont_common_causes:]}, copy=False)
        data = data.astype({wname: "category" for wname in common_causes[num_cont_common_causes:]}, copy=False)
    if num_discrete_effect_modifiers > 0 and not one_hot_encode:
        data = data.astype({emodname: "int64" for emodname in effect_modifiers[num_cont_effect_modifiers:]}, copy=False)
        data = data.astype(
            {emodname: "category" for emodname in effect_modifiers[num_cont_effect_modifiers:]}, copy=False
        )

    # Now specifying the corresponding graph strings
    dot_graph = create_dot_graph(treatments, outcome, common_causes, instruments, effect_modifiers, frontdoor_variables)
    # Now writing the gml graph
    gml_graph = create_gml_graph(treatments, outcome, common_causes, instruments, effect_modifiers, frontdoor_variables)
    ret_dict = {
        "df": data,
        "treatment_name": treatments,
        "outcome_name": outcome,
        "common_causes_names": common_causes,
        "instrument_names": instruments,
        "effect_modifier_names": effect_modifiers,
        "frontdoor_variables_names": frontdoor_variables,
        "dot_graph": dot_graph,
        "gml_graph": gml_graph,
        "ate": ate,
    }
    return ret_dict


def simple_iv_dataset(beta, num_samples, num_treatments=None, treatment_is_binary=True, outcome_is_binary=False):
    """Simple instrumental variable dataset with a single IV and a single confounder."""
    W, Z, c1, c2, cz = [None] * 5
    num_instruments = 1
    num_common_causes = 1

    beta = np.array(beta, dtype=np.float64, ndmin=1)
    if num_treatments is None:
        num_treatments = beta.size
    else:
        beta = np.resize(beta, num_treatments)

    c1 = np.random.uniform(0, 1, (num_common_causes, num_treatments))
    c2 = np.random.uniform(0, 1, num_common_causes)
    range_cz = 1 + np.max(np.absolute(beta))  # cz is much higher than c1 and c2
    cz = np.random.uniform(
        range_cz - (range_cz * 0.05), range_cz + (range_cz * 0.05), (num_instruments, num_treatments)
    )
    W = np.random.uniform(0, 1, (num_samples, num_common_causes))
    Z = np.random.normal(0, 1, (num_samples, num_instruments))
    t = np.random.normal(0, 1, (num_samples, num_treatments)) + Z @ cz + W @ c1
    if treatment_is_binary:
        t = np.vectorize(convert_to_binary)(t)

    def _compute_y(t, W, beta, c2):
        y = t @ beta + W @ c2
        return y

    y = _compute_y(t, W, beta, c2)

    # creating data frame
    data = np.column_stack((Z, W, t, y))
    treatments = [("v" + str(i)) for i in range(0, num_treatments)]
    outcome = "y"
    common_causes = [("W" + str(i)) for i in range(0, num_common_causes)]
    ate = np.mean(
        _compute_y(np.ones((num_samples, num_treatments)), W, beta, c2)
        - _compute_y(np.zeros((num_samples, num_treatments)), W, beta, c2)
    )
    instruments = [("Z" + str(i)) for i in range(0, num_instruments)]
    other_variables = None
    col_names = instruments + common_causes + treatments + [outcome]
    data = pd.DataFrame(data, columns=col_names)

    # Specifying the correct dtypes
    if treatment_is_binary:
        data = data.astype({tname: "bool" for tname in treatments}, copy=False)
    if outcome_is_binary:
        data = data.astype({outcome: "bool"}, copy=False)

    # Now specifying the corresponding graph strings
    dot_graph = create_dot_graph(treatments, outcome, common_causes, instruments)
    # Now writing the gml graph
    gml_graph = create_gml_graph(treatments, outcome, common_causes, instruments)
    ret_dict = {
        "df": data,
        "treatment_name": treatments,
        "outcome_name": outcome,
        "common_causes_names": common_causes,
        "instrument_names": instruments,
        "effect_modifier_names": None,
        "dot_graph": dot_graph,
        "gml_graph": gml_graph,
        "ate": ate,
    }
    return ret_dict


def create_dot_graph(treatments, outcome, common_causes, instruments, effect_modifiers=[], frontdoor_variables=[]):
    dot_graph = "digraph {"
    for currt in treatments:
        if len(frontdoor_variables) == 0:
            dot_graph += "{0}->{1};".format(currt, outcome)
        dot_graph += " ".join([v + "-> " + currt + ";" for v in common_causes])
        dot_graph += " ".join([v + "-> " + currt + ";" for v in instruments])
        dot_graph += " ".join([currt + "-> " + v + ";" for v in frontdoor_variables])

    dot_graph += " ".join([v + "-> " + outcome + ";" for v in common_causes])
    dot_graph += " ".join([v + "-> " + outcome + ";" for v in effect_modifiers])
    dot_graph += " ".join([v + "-> " + outcome + ";" for v in frontdoor_variables])
    dot_graph = dot_graph + "}"
    # Adding edges between common causes and the frontdoor mediator
    for v1 in common_causes:
        dot_graph += " ".join([v1 + "-> " + v2 + ";" for v2 in frontdoor_variables])
    return dot_graph


def create_gml_graph(treatments, outcome, common_causes, instruments, effect_modifiers=[], frontdoor_variables=[]):
    gml_graph = ("graph[directed 1" 'node[ id "{0}" label "{0}"]').format(outcome)

    gml_graph += " ".join(['node[ id "{0}" label "{0}"]'.format(v) for v in common_causes])
    gml_graph += " ".join(['node[ id "{0}" label "{0}"]'.format(v) for v in instruments])
    gml_graph += " ".join(['node[ id "{0}" label "{0}"]'.format(v) for v in frontdoor_variables])
    for currt in treatments:
        gml_graph += ('node[ id "{0}" label "{0}"]').format(currt)
        if len(frontdoor_variables) == 0:
            gml_graph += 'edge[source "{0}" target "{1}"]'.format(currt, outcome)
        gml_graph += " ".join(['edge[ source "{0}" target "{1}"]'.format(v, currt) for v in common_causes])
        gml_graph += " ".join(['edge[ source "{0}" target "{1}"]'.format(v, currt) for v in instruments])
        gml_graph += " ".join(['edge[ source "{0}" target "{1}"]'.format(currt, v) for v in frontdoor_variables])

    gml_graph = gml_graph + " ".join(['edge[ source "{0}" target "{1}"]'.format(v, outcome) for v in common_causes])
    gml_graph = gml_graph + " ".join(
        ['node[ id "{0}" label "{0}"] edge[ source "{0}" target "{1}"]'.format(v, outcome) for v in effect_modifiers]
    )
    gml_graph = gml_graph + " ".join(
        ['edge[ source "{0}" target "{1}"]'.format(v, outcome) for v in frontdoor_variables]
    )
    for v1 in common_causes:
        gml_graph = gml_graph + " ".join(
            ['edge[ source "{0}" target "{1}"]'.format(v1, v2) for v2 in frontdoor_variables]
        )
    gml_graph = gml_graph + "]"
    return gml_graph


def xy_dataset(num_samples, effect=True, num_common_causes=1, is_linear=True, sd_error=1):
    treatment = "Treatment"
    outcome = "Outcome"
    common_causes = ["w" + str(i) for i in range(num_common_causes)]
    time_var = "s"
    # Error terms
    E1 = np.random.normal(loc=0, scale=sd_error, size=num_samples)
    E2 = np.random.normal(loc=0, scale=sd_error, size=num_samples)

    S = np.random.uniform(0, 10, num_samples)
    T1 = 4 - (S - 3) * (S - 3)
    T1[S >= 5] = 0
    T2 = (S - 7) * (S - 7) - 4
    T2[S <= 5] = 0
    W0 = T1 + T2  # hidden confounder
    tterm, yterm = 0, 0
    if num_common_causes > 1:
        means = np.random.uniform(-1, 1, num_common_causes - 1)
        cov_mat = np.diag(np.ones(num_common_causes - 1))
        otherW = np.random.multivariate_normal(means, cov_mat, num_samples)
        c1 = np.random.uniform(0, 1, (otherW.shape[1], 1))
        c2 = np.random.uniform(0, 1, (otherW.shape[1], 1))
        tterm = (otherW @ c1)[:, 0]
        yterm = (otherW @ c2)[:, 0]

    if is_linear:
        V = 6 + W0 + tterm + E1
        Y = 6 + W0 + yterm + E2  # + (V-8)*(V-8)
        if effect:
            Y += V
        else:
            Y += 6 + W0
    else:
        V = 6 + W0 * W0 + tterm + E1
        Y = 6 + W0 * W0 + yterm + E2  # + (V-8)*(V-8)
        if effect:
            Y += V  # /20 # divide by 10 to scale the value of Y to be comparable to V
        else:
            Y += 6 + W0
    # else:
    #    V = 6 + W0 + tterm + E1
    #    Y = 12 + W0*W0 + W0*W0 + yterm + E2  # E2_new
    dat = {treatment: V, outcome: Y, common_causes[0]: W0, time_var: S}
    if num_common_causes > 1:
        for i in range(otherW.shape[1]):
            dat[common_causes[i + 1]] = otherW[:, i]
    data = pd.DataFrame(data=dat)
    ret_dict = {
        "df": data,
        "treatment_name": treatment,
        "outcome_name": outcome,
        "common_causes_names": common_causes,
        "time_val": time_var,
        "instrument_names": None,
        "dot_graph": None,
        "gml_graph": None,
        "ate": None,
    }
    return ret_dict


def create_discrete_column(num_samples, std_dev=1):
    # Generating a random normal distribution of integers
    x = np.arange(-5, 6)
    xU, xL = x + 0.5, x - 0.5
    prob = ss.norm.cdf(xU, scale=std_dev) - ss.norm.cdf(
        xL, scale=std_dev
    )  # probability of selecting a number x is p(x-0.5 < x < x+0.5) where x is a normal random variable with mean 0 and standard deviation std_dev
    prob = prob / prob.sum()  # normalize the probabilities so their sum is 1
    nums = choice(a=x, size=num_samples, p=prob)  # pick up an element
    return nums


def convert_continuous_to_discrete(arr):
    return arr.astype(int)


def generate_random_graph(n, max_iter=10):
    """
    Function to generate random Directed Acyclic Graph
    :param n: number of nodes in the graph
    :param max_iter: number of iterations to create graph

    :returns: Directed Acyclic Graph
    See: https://datascience.oneoffcoder.com/generate-random-bbn.html
    """
    g = get_simple_ordered_tree(n)
    for it in range(max_iter):
        i, j = get_random_node_pair(n)
        if g.has_edge(i, j) is True:
            del_edge(i, j, g)
        else:
            add_edge(i, j, g)
    return g


# Given a graph and a dictionary specifying the value type of the nodes in the graph
# (e.g. 'continuous', 'discrete', or 'binary'), verify that the dictionary fully specifies
# the value type for all notes in the graph.
def check_all_node_types_are_specified(graph, variable_type_dict):
    for node in list(graph.nodes):
        if node not in variable_type_dict.keys():
            raise ValueError(f"Graph node '{node}' is not present in variable_type_dict")
        if variable_type_dict[node] not in [CONTINUOUS, DISCRETE, BINARY]:
            raise ValueError(f"Graph node '{node}' does not have a valid value type in variable_type_dict")


# If the treatments are the only non-continuous variables in the graph, and the data is 'linear',
# then the exact ATE can be computed based on the graph weights.
def exact_ate_of_linear_graph(graph, treatments, outcome):
    ate = 0
    for treatment in treatments:
        for path in nx.all_simple_paths(graph, source=treatment, target=outcome):
            path_multiplier = 1
            if set(treatments).intersection(path[1:-1]):
                continue
            for a, b in zip(path, path[1:]):
                if graph[a][b]["weight"] is None:
                    raise ValueError(f"Missing edge weight from '{a}' to '{b}'")
                path_multiplier *= graph[a][b]["weight"]
            ate += path_multiplier
    return ate


def linear_dataset_from_graph(
    graph,
    treatments,
    outcome,
    treatments_are_binary=False,
    outcome_is_binary=False,
    variable_type_dict=None,
    num_samples=1000,
):
    """
    This function generates a random dataset given a graph plus a specification for the datatype of
    the variables in the graph (discrete, continuous, etc.).
    It models the variables linearly according to the relations in the graph.

    :param graph: A causal DAG giving the relationship between the variables
    :param treatments: The treatment variables we are measuring the effect of
    :param outcome: The outcome variable we are measuring the effect on
    :param treatments_are_binary: Indicates whether the treatment variables should be binary
    :param outcome_is_binary: Indicates whether the outcome variable should be binary
    :param variable_type_dict : An optional dictionary giving the data-type of all the variables in the graph.
    Keys in the dictionary are nodes in the graph, and values must be in ["discrete", "binary", "continuous"].
    Leave as None unless all nodes are specified; this argument is for the case where specifying the data type of the
    treatments and outcome is not enough.
    :param num_samples: Number of samples in the dataset
    :returns ret_dict : dictionary with information like dataframe, outcome, treatment, graph string and continuous, discrete and binary columns
    """
    # Set variable_type_dict based on given method parameters
    if variable_type_dict:
        check_all_node_types_are_specified(graph, variable_type_dict)
    else:
        variable_type_dict = {node: CONTINUOUS for node in graph.nodes}
        if treatments_are_binary:
            variable_type_dict.update({node: BINARY for node in treatments})
        if outcome_is_binary:
            variable_type_dict[outcome] = BINARY
    # Randomly assign edge weights
    for u, v in graph.edges():
        graph[u][v]["weight"] = np.random.uniform(0, 1)

    # This method computes the df, with an option to intervene
    def _compute_y(intervene=None):
        all_nodes = list(graph.nodes)
        all_nodes.sort()
        changed = dict()
        discrete_cols = []
        continuous_cols = []
        binary_cols = []
        for node in all_nodes:
            changed[node] = False
        df = pd.DataFrame()
        currset = list()
        counter = 0

        # Generating data for nodes which have no incoming edges
        for node in all_nodes:
            # If intervening, set treatment nodes
            if intervene == True and node in treatments:
                df[node] = np.ones(num_samples)
                continue
            elif intervene == False and node in treatments:
                df[node] = np.zeros(num_samples)
                continue

            # Otherwise, only set source nodes
            if graph.in_degree(node) == 0:
                dtype = variable_type_dict[node]
                if dtype == DISCRETE:
                    df[node] = create_discrete_column(num_samples)  # Generating discrete data
                    discrete_cols.append(node)
                elif dtype == CONTINUOUS:
                    df[node] = np.random.normal(0, 1, num_samples)  # Generating continuous data
                    continuous_cols.append(node)
                else:
                    nums = np.random.normal(0, 1, num_samples)
                    df[node] = np.vectorize(convert_to_binary)(nums)  # Generating binary data
                    discrete_cols.append(node)
                    binary_cols.append(node)
                successors = list(graph.successors(node))  # Storing immediate successors for next level data generation
                successors.sort()
                currset.extend(successors)
                changed[node] = True

        # "currset" variable currently has all the successors of the nodes which had no incoming edges
        while len(currset) > 0:
            cs = list()  # Variable to store immediate children of nodes present in "currset"
            for node in currset:
                predecessors = list(
                    graph.predecessors(node)
                )  # Getting all the parent nodes on which current "node" depends on
                if changed[node] == False and all(
                    changed[x] == True for x in predecessors
                ):  # Check if current "node" has not been processed yet and if all the parent nodes have been processed
                    successors = list(graph.successors(node))
                    successors.sort()
                    cs.extend(successors)  # Storing immediate children for next level data generation
                    changed[node] = True
                    if intervene == True and node in treatments:
                        continue
                    elif intervene == False and node in treatments:
                        continue
                    X_observed = df[predecessors].to_numpy()  # Using parent nodes data
                    c = np.array([graph[u][node]["weight"] for u in predecessors])
                    t_observed = (
                        np.random.normal(0, 1, num_samples) + X_observed @ c
                    )  # Using Linear Regression to generate data
                    dtype = variable_type_dict[node]
                    counter += 1
                    if dtype == DISCRETE:
                        df[node] = convert_continuous_to_discrete(t_observed)
                        discrete_cols.append(node)
                    elif dtype == CONTINUOUS:
                        df[node] = t_observed
                        continuous_cols.append(node)
                    else:
                        # nums = np.random.normal(0, 1, num_samples)
                        df[node] = np.vectorize(convert_to_binary)(t_observed)
                        discrete_cols.append(node)
                        binary_cols.append(node)
            currset = cs

        return df, continuous_cols, discrete_cols, binary_cols

    df, continuous_cols, discrete_cols, binary_cols = _compute_y()

    # Compute ATE:
    # If all non-treatment variables are continuous, then can be computed from the
    # graph directly
    if all(variable_type_dict[x] == CONTINUOUS for x in variable_type_dict.keys() if x not in treatments):
        ate = exact_ate_of_linear_graph(graph, treatments, outcome)
    else:  # Compute empirical ATE estimate
        df_treated, _, _, _ = _compute_y(intervene=True)
        df_untreated, _, _, _ = _compute_y(intervene=False)
        ate = np.mean(df_treated[outcome]) - np.mean(df_untreated[outcome])

    gml_str = "\n".join(nx.generate_gml(graph))
    ret_dict = {
        "df": df,
        "outcome_name": outcome,
        "treatment_name": treatments,
        "gml_graph": gml_str,
        "discrete_columns": discrete_cols,
        "continuous_columns": continuous_cols,
        "binary_columns": binary_cols,
        "ate_estimate": ate,
        "ate": ate,
    }
    return ret_dict


def dataset_from_random_graph(
    num_vars, num_samples=1000, prob_edge=0.3, random_seed=None, prob_type_of_data=(0.333, 0.333, 0.334)
):
    """
    This function generates a dataset with discrete and continuous kinds of variables.
    It creates a random graph and models the variables linearly according to the relations in the graph.

    :param num_vars: Number of variables in the dataset
    :param num_samples: Number of samples in the dataset
    :param prob_edge : Probability of an edge between two random nodes in a graph
    :param random_seed: Seed for generating random graph
    :param prob_type_of_data : 3-element tuple containing the probability of data being discrete, binary and continuous respectively.
    :returns ret_dict : dictionary with information like dataframe, outcome, treatment, graph string and continuous, discrete and binary columns
    """
    assert sum(list(prob_type_of_data)) == 1.0
    if random_seed is None:
        random_seed = np.random.randint(0, 1e6)
    np.random.seed(random_seed)
    DAG = generate_random_graph(n=num_vars)
    mapping = dict(zip(DAG, string.ascii_lowercase))
    DAG = nx.relabel_nodes(DAG, mapping)
    all_nodes = list(DAG.nodes)
    all_nodes.sort()
    num_nodes = len(all_nodes)
    random_numbers_array = np.random.rand(num_nodes)

    # Get treatments and outcomes
    outcome = None
    for node in all_nodes:
        if DAG.out_degree(node) == 0:
            outcome = node  # Node which has no successors is outcome
            break

    treatments = list()
    for node in all_nodes:
        if DAG.in_degree(node) > 0:
            children = list(DAG.successors(node))
            if outcome in children:
                treatments.append(node)  # Node which causes outcome is treatment

    # Build variable type dict
    variable_type_dict = {}
    for idx, node in enumerate(all_nodes):
        if random_numbers_array[idx] <= prob_type_of_data[0]:
            variable_type_dict[node] = DISCRETE
        elif random_numbers_array[idx] <= prob_type_of_data[0] + prob_type_of_data[1]:
            variable_type_dict[node] = CONTINUOUS
        else:
            variable_type_dict[node] = BINARY

    return linear_dataset_from_graph(
        DAG, treatments, outcome, variable_type_dict=variable_type_dict, num_samples=num_samples
    )


def partially_linear_dataset(
    beta,
    num_common_causes,
    num_unobserved_common_causes=0,
    strength_unobserved_confounding=1,
    num_samples=500,
    num_treatments=None,
    treatment_is_binary=True,
    treatment_is_category=False,
    outcome_is_binary=False,
    stochastic_discretization=True,
    num_discrete_common_causes=0,
    stddev_treatment_noise=1,
    stddev_outcome_noise=0,
    one_hot_encode=False,
    training_sample_size=10,
    random_state=0,
):
    assert not (treatment_is_binary and treatment_is_category)
    num_outcomes = 1
    beta = np.array(beta, dtype=np.float64, ndmin=1)
    if num_treatments is None:
        num_treatments = beta.size
    else:
        beta = np.resize(beta, num_treatments)
    num_cont_common_causes = num_common_causes - num_discrete_common_causes

    if num_common_causes > 0:
        range_c1 = 0.5 + np.max(np.absolute(beta)) * 0.5
        means = np.random.uniform(-1, 1, num_common_causes)
        cov_mat = np.diag(np.ones(num_common_causes))
        W = np.random.multivariate_normal(means, cov_mat, num_samples)
        W_with_dummy = convert_to_categorical(
            W, num_common_causes, num_discrete_common_causes, quantiles=[0.25, 0.5, 0.75], one_hot_encode=one_hot_encode
        )
        c1 = np.random.uniform(0, range_c1, (W_with_dummy.shape[1], num_treatments))
        # assuming that all unobserved common causes are numerical and are not affected by one hot encoding
        if num_unobserved_common_causes > 0:
            c1[:num_unobserved_common_causes] = c1[:num_unobserved_common_causes] * strength_unobserved_confounding
            for i in range(num_unobserved_common_causes, W_with_dummy.shape[1]):
                c1[i] = c1[i] * strength_unobserved_confounding / (2**i)

    # Creating a NN to simulate the nuisance function
    hidden_layer_arch = (50, 50, 50)
    neural_network = MLPRegressor(random_state=random_state, hidden_layer_sizes=hidden_layer_arch)
    x = np.random.randn(training_sample_size, W_with_dummy.shape[1])
    y = np.random.randn(training_sample_size, num_outcomes)
    neural_network.fit(x, np.ravel(y))

    T = np.random.normal(0, stddev_treatment_noise, (num_samples, num_treatments))
    T += W_with_dummy @ c1
    if treatment_is_binary:
        T = np.vectorize(convert_to_binary)(x=T)

    # strength of unobserved confounding
    strength_vec = np.ones(W_with_dummy.shape[1])
    strength_vec[:num_unobserved_common_causes] = (
        strength_vec[:num_unobserved_common_causes] * strength_unobserved_confounding
    )
    for i in range(num_unobserved_common_causes, W_with_dummy.shape[1]):
        strength_vec[i] = strength_vec[i] * strength_unobserved_confounding / (2**i)

    def _compute_y(T, W, beta, nn, stddev_outcome_noise):
        f_x = nn.predict(W)
        y = np.random.normal(0, stddev_outcome_noise, num_samples)
        y += T @ beta
        y += f_x
        if outcome_is_binary:
            y = np.vectorize(convert_to_binary)(y, stochastic_discretization)
        return y

    Y = _compute_y(T, W_with_dummy * strength_vec, beta, neural_network, stddev_outcome_noise)

    data = np.column_stack((T, Y))
    if num_common_causes > 0:
        data = np.column_stack((W_with_dummy, data))

    # Computing ATE
    T1 = np.ones((num_samples, num_treatments))
    T0 = np.zeros((num_samples, num_treatments))
    ate = np.mean(
        _compute_y(T1, W_with_dummy, beta, neural_network, stddev_outcome_noise)
        - _compute_y(T0, W_with_dummy, beta, neural_network, stddev_outcome_noise)
    )

    treatments = [("v" + str(i)) for i in range(0, num_treatments)]
    outcome = "y"
    common_causes = construct_col_names(
        "W", num_common_causes, num_discrete_common_causes, num_discrete_levels=4, one_hot_encode=one_hot_encode
    )
    col_names = common_causes + treatments + [outcome]
    data = pd.DataFrame(data, columns=col_names)
    # Specifying the correct dtypes
    if treatment_is_binary:
        data = data.astype({tname: "bool" for tname in treatments}, copy=False)
    elif treatment_is_category:
        data = data.astype({tname: "category" for tname in treatments}, copy=False)
    if outcome_is_binary:
        data = data.astype({outcome: "bool"}, copy=False)
    if num_discrete_common_causes > 0 and not one_hot_encode:
        data = data.astype({wname: "int64" for wname in common_causes[num_cont_common_causes:]}, copy=False)
        data = data.astype({wname: "category" for wname in common_causes[num_cont_common_causes:]}, copy=False)
    dot_graph = create_dot_graph(treatments, outcome, common_causes, instruments=[])
    # Now writing the gml graph
    gml_graph = create_gml_graph(treatments, outcome, common_causes, instruments=[])
    ret_dict = {
        "df": data,
        "treatment_name": treatments,
        "outcome_name": outcome,
        "common_causes_names": common_causes,
        "dot_graph": dot_graph,
        "gml_graph": gml_graph,
        "ate": ate,
    }
    return ret_dict


def lalonde_dataset() -> pd.DataFrame:
    """Downloads and returns the Lalonde dataset from https://users.nber.org/~rdehejia/nswdata2.html"""
    # The following code for loading the Lalonde dataset was copied from
    # https://github.com/wayfair/pylift/blob/5afc9088e96f25672423663f5c9b4bb889b4dfc0/examples/Lalonde/Lalonde_sample.ipynb?short_path=b1d451f#L94-L99).
    #
    # Copyright 2018, Wayfair, Inc.
    #
    # Redistribution and use in source and binary forms, with or without modification, are permitted provided that
    # the following conditions are met:
    #
    # 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the
    #    following disclaimer.
    #
    # 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the
    #    following disclaimer in the documentation and/or other materials provided with the distribution.
    #
    # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
    # WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
    # PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
    # DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
    # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    # CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
    # OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
    # DAMAGE.
    cols = ["treat", "age", "educ", "black", "hisp", "married", "nodegr", "re74", "re75", "re78"]
    control = pd.read_csv(
        "https://www.nber.org/~rdehejia/data/nswre74_control.txt", sep="\\s+", header=None, names=cols
    )
    treated = pd.read_csv(
        "https://www.nber.org/~rdehejia/data/nswre74_treated.txt", sep="\\s+", header=None, names=cols
    )
    lalonde = pd.concat([control, treated], ignore_index=True).astype({"treat": "bool"}, copy=False)
    lalonde["u74"] = np.where(lalonde["re74"] == 0, 1.0, 0.0)
    lalonde["u75"] = np.where(lalonde["re75"] == 0, 1.0, 0.0)
    return lalonde


def psid_dataset() -> pd.DataFrame:
    """Downloads and returns the PSID dataset from https://users.nber.org/~rdehejia/nswdata2.html

    This is a non-experimental comparison group constructed by Lalonde, consisting
    entirely of control observations.
    """
    # The following code is a slight modification of
    # https://github.com/wayfair/pylift/blob/5afc9088e96f25672423663f5c9b4bb889b4dfc0/examples/Lalonde/Lalonde_sample.ipynb?short_path=b1d451f#L94-L99).
    #
    # Copyright 2018, Wayfair, Inc.
    #
    # Redistribution and use in source and binary forms, with or without modification, are permitted provided that
    # the following conditions are met:
    #
    # 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the
    #    following disclaimer.
    #
    # 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the
    #    following disclaimer in the documentation and/or other materials provided with the distribution.
    #
    # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
    # WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
    # PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
    # DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
    # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    # CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
    # OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
    # DAMAGE.
    cols = ["treat", "age", "educ", "black", "hisp", "married", "nodegr", "re74", "re75", "re78"]
    psid = pd.read_csv("https://www.nber.org/~rdehejia/data/psid_controls.txt", sep="\\s+", header=None, names=cols)
    psid = psid.astype({"treat": "bool"}, copy=False)
    psid["u74"] = np.where(psid["re74"] == 0, 1.0, 0.0)
    psid["u75"] = np.where(psid["re75"] == 0, 1.0, 0.0)
    return psid


def sales_dataset(
    start_date: str = "2021-01-01",
    end_date: str = "2021-12-31",
    frequency: str = "d",
    num_shopping_events: int = 15,
    original_product_price: int = 1000,
    product_production_cost: int = 500,
    based_ad_spending: int = 1000,
    change_of_price: float = 1.0,
    change_of_demand: float = 1.25,
    page_visitor_factor: float = 1.0,
) -> pd.DataFrame:
    """
    Create a sales dataset based on a single product item with daily data.

    This closely follows the blog post: https://aws.amazon.com/blogs/opensource/root-cause-analysis-with-dowhy-an-open-source-python-library-for-causal-machine-learning/

    :param start_date: The starting date for the dataset, formatted as "YYYY-MM-DD". Default is "2021-01-01".
    :param end_date: The ending date for the dataset, formatted as "YYYY-MM-DD". Default is "2021-12-31".
    :param frequency: Frequency for the date range. Default is "d" (daily).
    :param num_shopping_events: Number of special shopping events. Default is 15.
    :param original_product_price: The initial price of the product. Default is 1000.
    :param product_production_cost: Cost of producing one unit of the product. Default is 500.
    :param based_ad_spending: Base spending on ad campaigns. Default is 1000.
    :param change_of_price: Factor by which the price changes. For example, a value of 0.9 means a 10% decrease. Default is 1.0.
    :param change_of_demand: Factor by which the demand changes with a change in price. See
                             https://en.wikipedia.org/wiki/Price_elasticity_of_demand for more information.
                             This influences the number of sold units. Default is 1.25.
    :param page_visitor_factor: A factor to adjust the number of page visits. Default is 1.0.

    :return: A dataframe containing columns related to sales data. The columns of the dataset are:
       - Shopping Event?: A binary value indicating whether a special shopping event took place.
       - Ad Spend: Spending on ad campaigns.
       - Page Views: Number of visits on the product detail page.
       - Unit Price: Price of the device, which could vary due to temporary discounts.
       - Sold Units: Number of sold units.
       - Revenue: Daily revenue.
       - Operational Cost: Daily operational expenses.
       - Profit: Daily profit.
    """
    shopping_event_col = "Shopping Event?"
    ad_spend_col = "Ad Spend"
    page_visit_col = "Page Views"
    price_col = "Unit Price"
    units_sold_col = "Sold Units"
    revenue_col = "Revenue"
    profit_col = "Profit"
    date_col = "Date"
    operation_col = "Operational Cost"

    days = pd.date_range(start=start_date, end=end_date, inclusive="both", freq=frequency)
    shopping_events = days[np.random.choice(days.shape[0], size=num_shopping_events, replace=False)]
    base_price = original_product_price * change_of_price

    df = pd.DataFrame({date_col: days})

    df[shopping_event_col] = df[date_col].isin(shopping_events)

    df[ad_spend_col] = (
        based_ad_spending
        + df[shopping_event_col] * uniform.rvs(loc=1000, scale=1000, size=df.shape[0])
        + (1 - df[shopping_event_col]) * uniform.rvs(loc=100, scale=400, size=df.shape[0])
    )

    df[page_visit_col] = (
        poisson.rvs(mu=10000 * page_visitor_factor, size=df.shape[0])
        + uniform.rvs(loc=5000 * page_visitor_factor, scale=5000, size=df.shape[0]) * df[shopping_event_col]
        + halfnorm.rvs(loc=0.5 * page_visitor_factor, scale=0.01, size=df.shape[0]) * df[ad_spend_col]
        + halfnorm.rvs(loc=1000 * page_visitor_factor, scale=100, size=df.shape[0])
    )
    df[page_visit_col] = df[page_visit_col].astype(int)

    df[price_col] = (
        base_price
        + uniform.rvs(loc=-200, scale=200, size=df.shape[0]) * df[shopping_event_col]
        + bernoulli.rvs(p=0.02, size=df.shape[0]) * uniform.rvs(loc=-20, scale=20, size=df.shape[0])
    )

    price_changes = 1 - df[price_col] / original_product_price
    demand_changes = price_changes * change_of_demand * (1 + price_changes)
    demand_changes[price_changes == 0] = 1
    price_changes[price_changes == 0] = 1

    df[units_sold_col] = [
        poisson.rvs(
            mu=demand_changes.iloc[i] / price_changes.iloc[i] * 0.2 * df[page_visit_col].iloc[i]
            + uniform.rvs(loc=100, scale=1000) * df[shopping_event_col].iloc[i]
        )
        for i in range(df.shape[0])
    ]
    df[units_sold_col] = df[units_sold_col].astype(int)

    df[revenue_col] = df[price_col] * df[units_sold_col]

    df[operation_col] = (
        df[ad_spend_col]
        + product_production_cost * df[units_sold_col]
        + halfnorm.rvs(loc=500000, scale=10, size=df.shape[0])
    )

    df[profit_col] = df[revenue_col] - df[operation_col]

    return df
