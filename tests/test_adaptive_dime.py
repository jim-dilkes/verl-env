"""Tests for AdaptiveDIME supplement ratio controller."""

from verl.trainer.ppo.adaptive_dime import AdaptiveDIME


def test_initial_supplement_is_max():
    ad = AdaptiveDIME(supplement_min=0.1, supplement_max=0.9, window_size=5, k=5.0)
    assert ad.current_supplement_prob == 0.9


def test_buffer_not_full_returns_initial():
    ad = AdaptiveDIME(supplement_min=0.1, supplement_max=0.9, window_size=5, k=5.0)
    for i in range(4):
        result = ad.update(float(i))
    assert result == 0.9  # buffer not yet full


def test_improving_rewards_lower_supplement():
    ad = AdaptiveDIME(supplement_min=0.1, supplement_max=0.9, window_size=5, k=5.0)
    for i in range(5):
        ad.update(float(i))  # steadily increasing: 0,1,2,3,4
    assert ad.current_supplement_prob < 0.5


def test_declining_rewards_higher_supplement():
    ad = AdaptiveDIME(supplement_min=0.1, supplement_max=0.9, window_size=5, k=5.0)
    for i in range(5):
        ad.update(float(4 - i))  # steadily decreasing: 4,3,2,1,0
    assert ad.current_supplement_prob > 0.5


def test_flat_rewards_mid_supplement():
    ad = AdaptiveDIME(supplement_min=0.1, supplement_max=0.9, window_size=5, k=5.0)
    for _ in range(5):
        ad.update(1.0)  # constant
    # Slope is 0 → sigmoid(0)=0.5 → supplement_min + 0.5*(max-min) = 0.5
    assert 0.45 < ad.current_supplement_prob < 0.55


def test_supplement_stays_in_bounds():
    ad = AdaptiveDIME(supplement_min=0.1, supplement_max=0.9, window_size=3, k=50.0)
    # Extreme positive slope
    for v in [0.0, 100.0, 200.0]:
        ad.update(v)
    assert ad.current_supplement_prob >= 0.1
    # Extreme negative slope
    ad2 = AdaptiveDIME(supplement_min=0.1, supplement_max=0.9, window_size=3, k=50.0)
    for v in [200.0, 100.0, 0.0]:
        ad2.update(v)
    assert ad2.current_supplement_prob <= 0.9


def test_get_no_supplement_prob():
    ad = AdaptiveDIME(supplement_min=0.1, supplement_max=0.9, window_size=5, k=5.0)
    assert abs(ad.get_no_supplement_prob() - 0.1) < 1e-9  # 1 - 0.9


def test_metrics_keys():
    ad = AdaptiveDIME(supplement_min=0.1, supplement_max=0.9, window_size=3, k=5.0)
    for i in range(3):
        ad.update(float(i))
    m = ad.get_metrics()
    assert "dime/adaptive_supplement_prob" in m
    assert "dime/adaptive_slope" in m
    assert "dime/adaptive_buffer_fill" in m
    assert m["dime/adaptive_buffer_fill"] == 1.0


if __name__ == "__main__":
    test_initial_supplement_is_max()
    test_buffer_not_full_returns_initial()
    test_improving_rewards_lower_supplement()
    test_declining_rewards_higher_supplement()
    test_flat_rewards_mid_supplement()
    test_supplement_stays_in_bounds()
    test_get_no_supplement_prob()
    test_metrics_keys()
    print("All AdaptiveDIME tests passed!")
