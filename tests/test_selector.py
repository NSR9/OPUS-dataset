import torch

from production.config import OpusConfig
from production.opus.selector import OpusSelector


def _make_features(n_local=8, n_layers=3, m=64):
    torch.manual_seed(0)
    c = {}
    p = {}
    for i in range(n_layers):
        name = f"layer_{i}"
        c[name] = torch.randn(n_local, m)
        p[name] = torch.randn(m)
    return c, p


def test_selector_is_deterministic_given_seed():
    cfg = OpusConfig(sketch_dim=64, selection_ratio=0.5, temperature=0.9, sketch_seed=123)
    c, p = _make_features(m=64)

    s1 = OpusSelector(cfg)
    r1 = s1.select(c, p, learning_rate=1e-3)

    s2 = OpusSelector(cfg)
    r2 = s2.select(c, p, learning_rate=1e-3)

    assert torch.equal(r1.selected_local_indices.cpu(), r2.selected_local_indices.cpu())


def test_selector_fallback_on_timeout():
    cfg = OpusConfig(sketch_dim=64, selection_ratio=0.5, temperature=0.9, sketch_seed=1)
    cfg.max_selector_time_s = 0.0
    c, p = _make_features(m=64)

    s = OpusSelector(cfg)
    r = s.select(c, p, learning_rate=1e-3)

    assert r.used_fallback
    assert r.selected_local_indices.numel() >= 1


def test_selector_sanitizes_nonfinite_features():
    cfg = OpusConfig(sketch_dim=32, selection_ratio=0.5, temperature=0.9, sketch_seed=4)
    c, p = _make_features(n_local=6, n_layers=2, m=32)
    c["layer_0"][0, 0] = float("nan")
    c["layer_1"][1, 1] = float("inf")
    p["layer_0"][2] = float("-inf")

    s = OpusSelector(cfg)
    r = s.select(c, p, learning_rate=1e-3)

    assert r.selected_local_indices.numel() >= 1
    assert r.selected_global_indices.numel() >= 1
    assert float(r.metrics["nonfinite_feature_values"]) > 0.0


def test_selector_handles_all_invalid_inputs_without_empty_selection():
    cfg = OpusConfig(sketch_dim=16, selection_ratio=0.5, temperature=0.9, sketch_seed=9)
    c = {"layer_0": torch.full((4, 16), float("nan"))}
    p = {"layer_0": torch.full((16,), float("nan"))}

    s = OpusSelector(cfg)
    r = s.select(c, p, learning_rate=1e-3)

    assert r.selected_local_indices.numel() >= 1
    assert r.selected_global_indices.numel() >= 1
    assert float(r.metrics["nonfinite_feature_values"]) > 0.0
