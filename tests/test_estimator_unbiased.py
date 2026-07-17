import random
from statistics import mean

from rescuecredit.estimators import PatchEMA, residual_estimate


def test_residual_estimator_unbiased():
    true_g0, mu, probability = 0.2, 0.7, 0.1
    rng = random.Random(1234)
    estimates = []
    for _ in range(100000):
        draw = int(rng.random() < probability)
        estimates.append(residual_estimate(mu, draw, probability, true_g0 if draw else None))
    assert abs(mean(estimates) - true_g0) < 0.02


def test_patch_ema_does_not_update_on_predict():
    ema = PatchEMA(beta=0.5, cold_start=0.25)
    assert ema.predict("p1") == 0.25
    assert ema.state_dict()["counts"] == {}
    ema.update("p1", 1.0)
    assert ema.predict("p1") == 0.625

