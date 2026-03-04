import torch
import torch.nn as nn

from production.config import OpusConfig
from production.opus import GhostCollector, AdamWPreconditionerView, OpusSelector
from recurrence_model_70b import MoEFFN


class _ToyMoEModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(
            [MoEFFN(d_model=16, d_hidden=8, num_experts=4, top_k=2, data_sparsity=0.5)]
        )
        self.head = nn.Linear(16, 8, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, aux = self.layers[0](x)
        z = self.head(y)
        return z.pow(2).mean() + aux


def test_moe_routed_capture_builds_selector_features():
    torch.manual_seed(7)

    model = _ToyMoEModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    preconditioner = AdamWPreconditionerView(optimizer)
    selector = OpusSelector(OpusConfig(sketch_dim=128, candidate_multiplier=2, proxy_batch_size=1))

    x = torch.randn(3, 4, 16, dtype=torch.float32, requires_grad=True)

    with GhostCollector(model) as collector:
        optimizer.zero_grad(set_to_none=True)
        loss = model(x)
        loss.backward()
        captures = collector.captures()
        moe_captures = collector.moe_captures()
        optimizer.zero_grad(set_to_none=True)

    preconditioner.refresh()
    candidate_features, proxy_features = selector.build_sketch_features(
        captures=captures,
        candidate_count=2,
        proxy_count=1,
        preconditioner=preconditioner,
        moe_captures=moe_captures,
        out_dtype=torch.float32,
    )

    assert moe_captures, "Expected at least one MoE routed capture"
    assert "layers.0.moe.W_gate" in candidate_features
    assert "layers.0.moe.W_up" in candidate_features
    assert "layers.0.moe.W_down" in candidate_features
    assert candidate_features["layers.0.moe.W_gate"].shape == (2, 128)
    assert proxy_features["layers.0.moe.W_gate"].shape == (128,)
    for tensor in list(candidate_features.values()) + list(proxy_features.values()):
        assert torch.isfinite(tensor).all()

