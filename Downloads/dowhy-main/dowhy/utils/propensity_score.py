import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from statsmodels.nonparametric.kernel_density import EstimatorSettings, KDEMultivariateConditional

from dowhy.utils.encoding import one_hot_encode


def propensity_of_treatment_score(data, covariates, treatment, model="logistic", variable_types=None):
    if model == "logistic":
        model = LogisticRegression(solver="lbfgs")
        data, covariates = binarize_discrete(data, covariates, variable_types)
        model = model.fit(data[covariates], data[treatment].values.ravel())
        scores = model.predict_proba(data[covariates])[:, 1]
        return scores
    else:
        raise NotImplementedError


def state_propensity_score(data, covariates, treatments, variable_types=None):
    if len(set(covariates).intersection(treatments)) != 0:
        raise Exception("Can't control for causal states. Remove treatment from covariates.")
    log_propensities = {}
    for i, treatment in enumerate(treatments):
        if variable_types[treatment] in ["b"]:
            log_propensities[treatment] = np.log(
                binary_treatment_model(data.copy(), covariates + treatments[i + 1 :], treatment, variable_types)
            )
        elif variable_types[treatment] in ["o", "u", "d"]:
            log_propensities[treatment] = np.log(
                categorical_treatment_model(data.copy(), covariates + treatments[i + 1 :], treatment, variable_types)
            )
        elif variable_types[treatment] in ["c"]:
            log_propensities[treatment] = np.log(
                continuous_treatment_model(data.copy(), covariates + treatments[i + 1 :], treatment, variable_types)
            )
        else:
            raise Exception(
                "Variable type {} for variable {} is not a recognized format type.".format(
                    variable_types[treatment], treatment
                )
            )
    scores = np.zeros(len(data))
    for treatment in treatments:
        scores += log_propensities[treatment]
    return np.exp(scores)


def binary_treatment_model(data, covariates, treatment, variable_types):
    data, covariates = binarize_discrete(data, covariates, variable_types)
    model = LogisticRegression(solver="lbfgs")
    model = model.fit(data[covariates], data[treatment])
    scores = model.predict_proba(data[covariates])
    scores = scores[range(len(scores)), data[treatment].values.astype(int)]
    return scores


def categorical_treatment_model(data, covariates, treatment, variable_types):
    data, covariates = binarize_discrete(data, covariates, variable_types)
    model = LogisticRegression(multi_class="ovr", solver="lbfgs")
    data[treatment], encoder = discrete_to_integer(data[treatment])
    model = model.fit(data[covariates], data[treatment])
    scores = model.predict_proba(data[covariates])
    scores = scores[range(len(data)), data[treatment].values.astype(int)]
    return scores


def continuous_treatment_model(data, covariates, treatment, variable_types):
    data, covariates = binarize_discrete(data, covariates, variable_types)
    if len(data) > 300 or len([treatment] + covariates) >= 3:
        defaults = EstimatorSettings(n_jobs=4, efficient=True)
    else:
        defaults = EstimatorSettings(n_jobs=-1, efficient=False)

    if "c" not in variable_types.values():
        bw = "cv_ml"
    else:
        bw = "normal_reference"

    indep_type = get_type_string(covariates, variable_types)
    dep_type = get_type_string([treatment], variable_types)

    model = KDEMultivariateConditional(
        endog=data[treatment],
        exog=data[covariates],
        dep_type="".join(dep_type),
        indep_type="".join(indep_type),
        bw=bw,
        defaults=defaults,
    )
    scores = model.pdf(endog_predict=data[treatment], exog_predict=data[covariates])
    return scores


def get_type_string(variables, variable_types):
    var_types = []
    for variable in variables:
        if variable_types[variable] in ["b", "d", "o", "u"]:
            if variable_types[variable] in ["o", "u"]:
                var_types.append(variable_types[variable])
            else:
                var_types.append("u")
        elif variable_types[variable] in ["c"]:
            var_types.append("c")
        else:
            raise Exception(
                "Variable type {} for variable {} not a recognized type.".format(variable_types[variable], variable)
            )
    return "".join(var_types)


def binarize_discrete(data, covariates, variable_types):
    to_remove = []
    if variable_types:
        for variable in covariates:
            variable_type = variable_types[variable]
            # variable_type:
            #  A dictionary containing the variable's names and types. 'c' for continuous, 'o'
            #  for ordered, 'd' for discrete, and 'u' for unordered discrete.
            if variable_type in ["d", "o", "u"]:
                # [] notation to retain DataFrame rather than Series.
                # For one_hot_encode type must be categorical, or it won't encode.
                variable_data = data.loc[:, [variable]].astype(str)
                dummies, _ = one_hot_encode(variable_data)  # Original impl. pd.get_dummies, drop_first default is False
                dummies.columns = [variable + str(col) for col in dummies.columns]
                dummies = dummies[dummies.columns[:-1]]
                covariates += list(dummies.columns)
                for var_name in dummies.columns:
                    variable_types[var_name] = "b"
                data = pd.concat((data, dummies), axis=1)
                to_remove.append(variable)
    for variable in to_remove:
        covariates.remove(variable)
        del data[variable]
    return data, covariates


def discrete_to_integer(discrete):
    encoder = LabelEncoder()
    discrete = encoder.fit_transform(discrete)
    return discrete, encoder
