"""Tests for AdaptiveICE supplement ratio controller."""

from verl.trainer.ppo.adaptive_ice import AdaptiveICE


def test_initial_supplement_is_min():
    ad = AdaptiveICE(supplement_min=0.1, supplement_max=0.9, window_size=5, k=5.0, inflection=0.5)
    assert ad.current_supplement_prob == 0.1


def test_buffer_not_full_returns_initial():
    ad = AdaptiveICE(supplement_min=0.1, supplement_max=0.9, window_size=5, k=5.0, inflection=0.5)
    for i in range(4):
        result = ad.update(float(i))
    assert result == 0.1  # buffer not yet full, stays at min


def test_improving_rewards_lower_supplement():
    ad = AdaptiveICE(supplement_min=0.1, supplement_max=0.9, window_size=5, k=5.0, inflection=0.5)
    for i in range(5):
        ad.update(float(i))  # steadily increasing: 0,1,2,3,4
    assert ad.current_supplement_prob < 0.5


def test_declining_rewards_higher_supplement():
    ad = AdaptiveICE(supplement_min=0.1, supplement_max=0.9, window_size=5, k=5.0, inflection=0.5)
    for i in range(5):
        ad.update(float(4 - i))  # steadily decreasing: 4,3,2,1,0
    assert ad.current_supplement_prob > 0.5


def test_flat_rewards_high_supplement():
    """With inflection=0.5, slope=0 is below inflection → high supplement."""
    ad = AdaptiveICE(supplement_min=0.0, supplement_max=0.875, window_size=5, k=5.0, inflection=0.5)
    for _ in range(5):
        ad.update(1.0)  # constant → slope=0, well below inflection
    assert ad.current_supplement_prob > 0.7


def test_inflection_zero_gives_mid_on_flat():
    """With inflection=0, slope=0 → sigmoid(0)=0.5 → mid supplement (original behaviour)."""
    ad = AdaptiveICE(supplement_min=0.1, supplement_max=0.9, window_size=5, k=5.0, inflection=0.0)
    for _ in range(5):
        ad.update(1.0)
    assert 0.45 < ad.current_supplement_prob < 0.55


def test_supplement_stays_in_bounds():
    ad = AdaptiveICE(supplement_min=0.1, supplement_max=0.9, window_size=3, k=50.0, inflection=0.5)
    # Extreme positive slope
    for v in [0.0, 100.0, 200.0]:
        ad.update(v)
    assert ad.current_supplement_prob >= 0.1
    # Extreme negative slope
    ad2 = AdaptiveICE(supplement_min=0.1, supplement_max=0.9, window_size=3, k=50.0, inflection=0.5)
    for v in [200.0, 100.0, 0.0]:
        ad2.update(v)
    assert ad2.current_supplement_prob <= 0.9


def test_get_no_supplement_prob():
    ad = AdaptiveICE(supplement_min=0.1, supplement_max=0.9, window_size=5, k=5.0, inflection=0.5)
    assert abs(ad.get_no_supplement_prob() - 0.9) < 1e-9  # 1 - 0.1 (starts at min)


def test_metrics_keys():
    ad = AdaptiveICE(supplement_min=0.1, supplement_max=0.9, window_size=3, k=5.0, inflection=0.5)
    for i in range(3):
        ad.update(float(i))
    m = ad.get_metrics()
    assert "ice/adaptive_supplement_prob" in m
    assert "ice/adaptive_slope" in m
    assert "ice/adaptive_inflection" in m
    assert "ice/adaptive_buffer_fill" in m
    assert m["ice/adaptive_buffer_fill"] == 1.0
    assert m["ice/adaptive_inflection"] == 0.5


if __name__ == "__main__":
    test_initial_supplement_is_min()
    test_buffer_not_full_returns_initial()
    test_improving_rewards_lower_supplement()
    test_declining_rewards_higher_supplement()
    test_flat_rewards_high_supplement()
    test_inflection_zero_gives_mid_on_flat()
    test_supplement_stays_in_bounds()
    test_get_no_supplement_prob()
    test_metrics_keys()
    print("All AdaptiveICE tests passed!")
