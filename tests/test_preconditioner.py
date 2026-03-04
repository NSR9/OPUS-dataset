import math

import torch

from production.opus.preconditioner import AdamWPreconditionerView


def test_adamw_preconditioner_matches_formula():
    p = torch.nn.Parameter(torch.zeros(2, 3))
    opt = torch.optim.AdamW([p], lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)

    # Populate optimizer state explicitly.
    state = opt.state[p]
    state["exp_avg_sq"] = torch.full_like(p, 4.0)
    state["step"] = torch.tensor(10.0)

    view = AdamWPreconditionerView(opt)
    pre = view.get(p)

    step = 10.0
    beta1, beta2 = 0.9, 0.95
    lr, eps = 1e-3, 1e-8
    c_t = lr * (1.0 - beta1) / (1.0 - beta1 ** step)
    v_hat = 4.0 / (1.0 - beta2 ** step)
    expected = c_t / (math.sqrt(v_hat) + eps)

    assert torch.allclose(pre, torch.full_like(pre, expected), atol=1e-8, rtol=1e-6)


def test_preconditioner_fallback_when_state_missing():
    p = torch.nn.Parameter(torch.zeros(2, 2))
    opt = torch.optim.AdamW([p], lr=2e-4, betas=(0.9, 0.99), eps=1e-8)
    view = AdamWPreconditionerView(opt)
    out = view.get(p)
    expected = 2e-4 * (1 - 0.9)
    assert torch.allclose(out, torch.full_like(out, expected))


def test_preconditioner_sanitizes_nonfinite_moment_state():
    p = torch.nn.Parameter(torch.zeros(2, 2))
    opt = torch.optim.AdamW([p], lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)

    state = opt.state[p]
    state["exp_avg_sq"] = torch.tensor([[float("nan"), -1.0], [float("inf"), 4.0]])
    state["step"] = torch.tensor(float("nan"))

    view = AdamWPreconditionerView(opt)
    out = view.get(p)
    assert bool(torch.isfinite(out).all())


def test_preconditioner_get_slices_matches_individual_get_slice():
    p = torch.nn.Parameter(torch.zeros(4, 3, 2))
    opt = torch.optim.AdamW([p], lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    state = opt.state[p]
    state["exp_avg_sq"] = torch.arange(p.numel(), dtype=torch.float32).view_as(p) + 1.0
    state["step"] = torch.tensor(5.0)

    view = AdamWPreconditionerView(opt)
    idx = torch.tensor([3, 1, 2], dtype=torch.long)
    batch = view.get_slices(p, idx)
    single = torch.stack([view.get_slice(p, int(i)) for i in idx.tolist()], dim=0)
    assert torch.allclose(batch, single, atol=1e-8, rtol=1e-6)


def test_preconditioner_strict_shard_mode_zeros_missing_state():
    p = torch.nn.Parameter(torch.zeros(2, 2))
    opt = torch.optim.AdamW([p], lr=2e-4, betas=(0.9, 0.99), eps=1e-8)
    view = AdamWPreconditionerView(opt, strict_shard_only=True)
    out = view.get(p)
    assert torch.allclose(out, torch.zeros_like(out))
