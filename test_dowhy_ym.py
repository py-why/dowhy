from dowhy import CausalModel
import numpy as np
import pickle

with open('data.pkl', 'rb') as f:
    data = pickle.load(f)

model = CausalModel(
    data=data,
    treatment='treatment_record',
    outcome='outcome_record',
    common_causes=['drug3','drug10','drug8','d2', 'd5', 'M','F','drug3_date','drug10_date',
                   'drug8_date', 'd2_date', 'd5_date', 'age']
    )

identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

metric = 'all'

causal_estimate = model.estimate_effect(identified_estimand,
        method_name='backdoor.propensity_score_matching', target_units=metric)

print(causal_estimate.value)