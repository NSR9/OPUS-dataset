import pytest
import torch

from opus.countsketch import TensorSketchOperator
from opus.layer_state import LayerConfig, LayerSketchState
from opus.preconditioner import compute_factored_preconditioner
from opus.scorer import OPUSScorer

# -----------------------------------------------------------------------
# TensorSketchOperator
# -----------------------------------------------------------------------


class TestTensorSketchOperator:
    def test_output_shape_unbatched(self):
        op = TensorSketchOperator(
            d_in=64, d_out=128, sketch_dim=256, device=torch.device("cpu")
        )
        a = torch.randn(64)
        b = torch.randn(128)
        out = op.sketch(a, b)
        assert out.shape == (256,)

    def test_output_shape_batched(self):
        op = TensorSketchOperator(
            d_in=64, d_out=128, sketch_dim=256, device=torch.device("cpu")
        )
        a = torch.randn(16, 64)
        b = torch.randn(16, 128)
        out = op.sketch(a, b)
        assert out.shape == (16, 256)

    def test_batched_matches_unbatched(self):
        """Each row of the batched output should match the corresponding unbatched call."""
        op = TensorSketchOperator(
            d_in=32, d_out=64, sketch_dim=128, device=torch.device("cpu"), seed=0
        )
        a = torch.randn(8, 32)
        b = torch.randn(8, 64)
        batched = op.sketch(a, b)

        for i in range(8):
            single = op.sketch(a[i], b[i])
            assert torch.allclose(batched[i], single, atol=1e-5), (
                f"Mismatch at index {i}"
            )

    def test_inner_product_preservation(self):
        torch.manual_seed(0)
        d_in, d_out, m = 64, 128, 4096
        op = TensorSketchOperator(d_in, d_out, m, device=torch.device("cpu"), seed=42)

        true_ips = []
        sketch_ips = []
        for _ in range(20):
            a1, a2 = torch.randn(d_in), torch.randn(d_in)
            b1, b2 = torch.randn(d_out), torch.randn(d_out)

            true_ip = torch.dot(b1, b2).item() * torch.dot(a1, a2).item()
            sketch_ip = torch.dot(op.sketch(a1, b1), op.sketch(a2, b2)).item()
            true_ips.append(abs(true_ip))
            sketch_ips.append(abs(true_ip - sketch_ip))

        mean_error = sum(sketch_ips) / len(sketch_ips)
        mean_scale = sum(true_ips) / len(true_ips)
        relative_error = mean_error / (mean_scale + 1e-8)

        assert relative_error < 0.5, (
            f"Mean relative inner product error too large: {relative_error:.4f}"
        )

    def test_different_seeds_give_different_sketches(self):
        a = torch.randn(32)
        b = torch.randn(64)
        op1 = TensorSketchOperator(32, 64, 128, torch.device("cpu"), seed=0)
        op2 = TensorSketchOperator(32, 64, 128, torch.device("cpu"), seed=1)
        assert not torch.allclose(op1.sketch(a, b), op2.sketch(a, b))


# -----------------------------------------------------------------------
# compute_factored_preconditioner
# -----------------------------------------------------------------------


class TestFactoredPreconditioner:
    def test_output_shapes(self):
        d_out, d_in = 128, 64
        v = torch.rand(d_out, d_in) + 1e-4
        u, w = compute_factored_preconditioner(v)
        assert u.shape == (d_out,)
        assert w.shape == (d_in,)

    def test_positivity(self):
        v = torch.rand(64, 32) + 1e-4
        u, w = compute_factored_preconditioner(v)
        assert (u > 0).all()
        assert (w > 0).all()

    def test_larger_v_gives_smaller_preconditioner(self):
        """Higher second moment -> smaller preconditioner scale (more cautious step)."""
        v_small = torch.ones(16, 16) * 0.01
        v_large = torch.ones(16, 16) * 100.0
        u_small, _ = compute_factored_preconditioner(v_small)
        u_large, _ = compute_factored_preconditioner(v_large)
        assert (u_small > u_large).all()

    def test_eps_prevents_division_by_zero(self):
        v = torch.zeros(32, 16)  # all zeros — would blow up without eps
        u, w = compute_factored_preconditioner(v, eps=1e-8)
        assert torch.isfinite(u).all()
        assert torch.isfinite(w).all()


# -----------------------------------------------------------------------
# LayerSketchState
# -----------------------------------------------------------------------


class TestLayerSketchState:
    def _make_state(self, d_in=32, d_out=64, m=256):
        config = LayerConfig(name="test", d_in=d_in, d_out=d_out)
        return LayerSketchState(config, sketch_dim=m, device=torch.device("cpu"))

    def test_proxy_sketch_shape(self):
        state = self._make_state()
        a = torch.randn(8, 32)
        b = torch.randn(8, 64)
        state.compute_proxy_sketch(a, b)
        assert state.proxy_sketch is not None
        assert state.proxy_sketch.shape == (256,)

    def test_candidate_sketch_shape(self):
        state = self._make_state()
        N = 16
        a = torch.randn(N, 32)
        b = torch.randn(N, 64)
        v = torch.rand(64, 32) + 1e-4
        out = state.compute_candidate_sketches(a, b, v)
        assert out.shape == (N, 256)

    def test_history_accumulates(self):
        state = self._make_state(m=128)
        sketch = torch.ones(128)
        state.update_history(sketch)
        state.update_history(sketch)
        assert torch.allclose(state.history, torch.full((128,), 2.0))

    def test_reset_clears_state(self):
        state = self._make_state()
        state.compute_proxy_sketch(torch.randn(4, 32), torch.randn(4, 64))
        state.update_history(torch.ones(256))
        state.reset()
        assert state.proxy_sketch is None
        assert state.history.sum().item() == 0.0


# -----------------------------------------------------------------------
# OPUSScorer
# -----------------------------------------------------------------------


class TestOPUSScorer:
    def _make_scorer(self, sketch_dim=256, temperature=0.9, eta=1.0):
        return OPUSScorer(
            sketch_dim=sketch_dim,
            temperature=temperature,
            eta=eta,
            device=torch.device("cpu"),
        )

    def _register_two_layers(self, scorer, d_in1=32, d_out1=64, d_in2=16, d_out2=32):
        scorer.register_layer("layer1", d_in=d_in1, d_out=d_out1, seed=42)
        scorer.register_layer("layer2", d_in=d_in2, d_out=d_out2, seed=99)

    def test_register_layer(self):
        scorer = self._make_scorer()
        scorer.register_layer("fc", d_in=16, d_out=32)
        assert "fc" in scorer.layer_states

    def test_full_pipeline_selection_count(self):
        scorer = self._make_scorer()
        self._register_two_layers(scorer)

        N, K_proxy, K = 32, 8, 16
        scorer.update_proxy_sketch(
            "layer1", torch.randn(K_proxy, 32), torch.randn(K_proxy, 64)
        )
        scorer.update_proxy_sketch(
            "layer2", torch.randn(K_proxy, 16), torch.randn(K_proxy, 32)
        )
        scorer.update_candidate_sketches(
            "layer1", torch.randn(N, 32), torch.randn(N, 64), torch.rand(64, 32) + 1e-4
        )
        scorer.update_candidate_sketches(
            "layer2", torch.randn(N, 16), torch.randn(N, 32), torch.rand(32, 16) + 1e-4
        )

        selected = scorer.run_selection(K)

        assert len(selected) == K
        assert len(set(selected)) == K  # no duplicates
        assert all(0 <= i < N for i in selected)  # valid indices

    def test_reset_clears_candidate_state(self):
        scorer = self._make_scorer()
        scorer.register_layer("fc", d_in=16, d_out=32)
        scorer.update_proxy_sketch("fc", torch.randn(4, 16), torch.randn(4, 32))
        scorer.update_candidate_sketches(
            "fc", torch.randn(8, 16), torch.randn(8, 32), torch.rand(32, 16) + 1e-4
        )

        scorer.reset_step()

        assert scorer._n_candidates is None
        assert len(scorer._candidate_sketches) == 0
        assert scorer.layer_states["fc"].proxy_sketch is None

    def test_unregistered_layer_raises(self):
        scorer = self._make_scorer()
        with pytest.raises(AssertionError, match="not registered"):
            scorer.update_proxy_sketch("ghost", torch.randn(4, 16), torch.randn(4, 32))

    def test_missing_proxy_sketch_raises(self):
        scorer = self._make_scorer()
        scorer.register_layer("fc", d_in=16, d_out=32)
        scorer.update_candidate_sketches(
            "fc", torch.randn(4, 16), torch.randn(4, 32), torch.rand(32, 16) + 1e-4
        )
        with pytest.raises(AssertionError, match="Proxy sketch not computed"):
            scorer.compute_utilities()

    def test_utility_scores_shape(self):
        scorer = self._make_scorer()
        scorer.register_layer("fc", d_in=16, d_out=32)
        scorer.update_proxy_sketch("fc", torch.randn(4, 16), torch.randn(4, 32))
        scorer.update_candidate_sketches(
            "fc", torch.randn(8, 16), torch.randn(8, 32), torch.rand(32, 16) + 1e-4
        )
        utilities = scorer.compute_utilities()
        assert utilities.shape == (8,)

    def test_similar_candidate_scores_higher(self):
        """
        A candidate whose gradient direction matches the proxy should score
        higher than a random candidate, under an identity-like preconditioner.
        """
        torch.manual_seed(42)
        scorer = OPUSScorer(
            sketch_dim=2048, temperature=0.9, eta=1.0, device=torch.device("cpu")
        )
        scorer.register_layer("fc", d_in=64, d_out=128, global_seed=0)

        K_proxy = 8
        proxy_a = torch.randn(K_proxy, 64)
        proxy_b = torch.randn(K_proxy, 128)
        scorer.update_proxy_sketch("fc", proxy_a, proxy_b)

        # Candidate 0: aligned with proxy mean direction
        # Candidate 1: random (likely misaligned)
        mean_a = proxy_a.mean(0, keepdim=True)
        mean_b = proxy_b.mean(0, keepdim=True)
        cand_a = torch.cat([mean_a, torch.randn(1, 64)], dim=0)
        cand_b = torch.cat([mean_b, torch.randn(1, 128)], dim=0)

        # Uniform v -> u ≈ w ≈ const, so preconditioning ≈ identity scaling
        v = torch.ones(128, 64)
        scorer.update_candidate_sketches("fc", cand_a, cand_b, v)

        utilities = scorer.compute_utilities()
        assert utilities[0] > utilities[1], (
            f"Aligned candidate should score higher: {utilities[0]:.4f} vs {utilities[1]:.4f}"
        )

    def test_redundancy_penalty_reduces_utility(self):
        """
        After selecting a candidate, its utility should decrease due to the
        redundancy penalty, even if it scores highest initially.
        """
        torch.manual_seed(0)
        scorer = OPUSScorer(
            sketch_dim=512, temperature=0.9, eta=1.0, device=torch.device("cpu")
        )
        scorer.register_layer("fc", d_in=32, d_out=64, global_seed=0)

        scorer.update_proxy_sketch("fc", torch.randn(4, 32), torch.randn(4, 64))
        scorer.update_candidate_sketches(
            "fc", torch.randn(8, 32), torch.randn(8, 64), torch.rand(64, 32) + 1e-4
        )

        utilities_before = scorer.compute_utilities().clone()
        best_idx = int(utilities_before.argmax().item())

        scorer.update_history(best_idx)
        utilities_after = scorer.compute_utilities()

        assert utilities_after[best_idx] < utilities_before[best_idx], (
            "Utility of selected candidate should decrease after history update"
        )
