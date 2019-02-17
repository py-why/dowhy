from sklearn.linear_model import LogisticRegression
from pandas import get_dummies


def propensity_score(covariates, treatment, method='logistic', variable_types=None):
    if method == 'logistic':
        model = LogisticRegression()
        covariates = binarize_discrete(covariates, variable_types)
        model = model.fit(covariates, treatment)
        scores = model.predict_proba(covariates)[:, 1]
        return scores
    else:
        raise NotImplementedError


def binarize_discrete(covariates, variable_types):
    if variable_types:
        for variable, variable_type in variable_types.items():
            if variable_type in ['d', 'o', 'u']:
                dummies = get_dummies(covariates[variable])
                dummies = dummies[dummies.columns[:-1]]
                del covariates[variable]
                covariates[dummies.columns] = dummies
    return covariates