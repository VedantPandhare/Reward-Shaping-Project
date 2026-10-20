import pandas as pd

from rewardlab.detector import HackingDetector, return_landing_divergence


def ep(truncated, length, custom, total):
    return {"raw_return": 0, "shaped_return": 0, "length": length, "truncated": truncated,
            "custom_sum": custom, "abs_total": total}


def test_all_timeouts_with_bonus_flagged():
    det = HackingDetector(window=50)
    for _ in range(50):
        det.add_episode(ep(True, 1000, 900.0, 1000.0))
    m = det.metrics()
    assert m["timeout_rate"] == 1.0 and m["mean_length"] == 1000
    assert m["custom_share"] > 0.5
    assert det.is_flagged()


def test_normal_landings_not_flagged():
    det = HackingDetector(window=50)
    for _ in range(50):
        det.add_episode(ep(False, 250, 0.0, 400.0))
    assert not det.is_flagged()


def test_timeouts_without_custom_term_not_flagged():
    # baseline that merely hovers: timeout but no custom bonus -> not "reward hacking"
    det = HackingDetector(window=50)
    for _ in range(50):
        det.add_episode(ep(True, 1000, 0.0, 300.0))
    assert not det.is_flagged()


def test_small_telescoping_shaping_not_flagged():
    det = HackingDetector(window=50)
    for _ in range(50):
        det.add_episode(ep(False, 300, 3.0, 400.0))
    assert not det.is_flagged()


def test_thresholds_configurable():
    det = HackingDetector(window=10, timeout_rate_threshold=0.9, custom_share_threshold=0.1)
    for i in range(10):
        det.add_episode(ep(i < 5, 500, 200.0, 400.0))
    assert not det.is_flagged()  # timeout rate 0.5 < 0.9


def test_analyze_dataframe():
    df = pd.DataFrame([ep(True, 1000, 900.0, 1000.0)] * 20)
    assert HackingDetector.analyze(df)["flagged"]


def test_window_rolls():
    det = HackingDetector(window=5)
    for _ in range(5):
        det.add_episode(ep(True, 1000, 900.0, 1000.0))
    for _ in range(5):
        det.add_episode(ep(False, 200, 0.0, 300.0))
    assert not det.is_flagged()


def test_divergence():
    assert return_landing_divergence([1, 2, 3, 4], [0.5, 0.4, 0.3, 0.1])
    assert not return_landing_divergence([1, 2, 3, 4], [0.1, 0.3, 0.5, 0.7])
