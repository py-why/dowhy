import networkx as nx
import numpy as np
import pandas as pd
import pytest

from dowhy import gcm
from dowhy.gcm._noise import noise_samples_of_ancestors
from dowhy.gcm.influence import intrinsic_causal_influence_sample
from dowhy.gcm.ml import create_linear_regressor, create_linear_regressor_with_given_parameters
from dowhy.gcm.shapley import ShapleyApproximationMethods, ShapleyConfig


@pytest.mark.parametrize("prediction_mode", ["exact", "linear", "approx"])
@pytest.mark.parametrize("background_variant", ["supplied", "permuted", "reordered", "shorter", "generated"])
def test_sample_influence_preserves_training_pairs_with_custom_background(
    prediction_mode, background_variant, monkeypatch
):
    # X = N_X, Y = 2 X + N_Y with independent, balanced +/-1 noises.
    model = gcm.InvertibleStructuralCausalModel(nx.DiGraph([("X", "Y")]))
    model.set_causal_mechanism("X", gcm.EmpiricalDistribution())
    model.set_causal_mechanism(
        "Y", gcm.AdditiveNoiseModel(create_linear_regressor_with_given_parameters(np.array([2.0])))
    )
    x = np.array([-1.0, -1.0, 1.0, 1.0])
    noise_y = np.array([-1.0, 1.0, -1.0, 1.0])
    gcm.fit(model, pd.DataFrame({"X": x, "Y": 2 * x + noise_y}))

    seed, num_training_samples = 812, 400
    np.random.seed(seed)
    _, generated_noise = noise_samples_of_ancestors(model, "Y", num_training_samples)
    # This legal background reverses the signs if incorrectly paired with the
    # outcomes generated from the original noise draws during model training.
    background = -generated_noise
    if background_variant == "permuted":
        background = background.sample(frac=1, random_state=99)
    elif background_variant == "reordered":
        background = background[["Y", "X"]]
    elif background_variant == "shorter":
        background = background.iloc[:-1]
    elif background_variant == "generated":
        background = generated_noise

    predictor = create_linear_regressor() if prediction_mode == "linear" else prediction_mode
    if prediction_mode == "approx":
        # Isolate training and attribution from automatic model-selection cost.
        monkeypatch.setattr(gcm.auto, "select_model", lambda *args: (create_linear_regressor(), None))

    np.random.seed(seed)
    result = intrinsic_causal_influence_sample(
        model,
        "Y",
        pd.DataFrame({"X": [1.0], "Y": [3.0]}),
        noise_feature_samples=None if background_variant == "generated" else background,
        prediction_model=predictor,
        num_noise_feature_samples=num_training_samples,
        shapley_config=ShapleyConfig(approximation_method=ShapleyApproximationMethods.EXACT, n_jobs=1),
    )[0]

    # For this additive SCM the contributions follow directly from the baseline
    # noise (1, 1) and background means, regardless of row or column order.
    assert result[("X", "Y")] == pytest.approx(2 * (1 - background["X"].mean()))
    assert result[("Y", "Y")] == pytest.approx(1 - background["Y"].mean())
