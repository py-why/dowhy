from sklearn.linear_model import LogisticRegression


def propensity_score(covariates, treatment, method='logistic', variable_types=None):
    if method == 'logistic':
        model = LogisticRegression()
        model = model.fit(covariates, treatment)
        scores = model.predict_proba(covariates)[:, 1]
        return scores
    else:
        raise NotImplementedError
