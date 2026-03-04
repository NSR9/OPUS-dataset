import os
import tempfile

import torch
from torch.utils.data import DataLoader, TensorDataset

from production.opus.proxy import (
    BenchProxyProvider,
    RandomInDistributionProxyProvider,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bench_tensor(n=20, seq_len=64):
    """Create a [N, L] token tensor and save to a temp file."""
    tokens = torch.arange(n * seq_len).view(n, seq_len)
    fd, path = tempfile.mkstemp(suffix=".pt")
    os.close(fd)
    torch.save(tokens, path)
    return path, tokens


def _make_loader(n=12, seq_len=32, batch_size=4):
    """Create a DataLoader yielding {"input_ids": tensor}."""
    ids = torch.arange(n * seq_len).view(n, seq_len)
    ds = TensorDataset(ids)
    # wrap in a collate that returns {"input_ids": ...}
    def collate(batch):
        return {"input_ids": torch.stack([b[0] for b in batch])}
    return DataLoader(ds, batch_size=batch_size, collate_fn=collate)


# ===========================================================================
# BenchProxyProvider tests
# ===========================================================================

class TestBenchProxyProvider:

    def test_output_shape(self):
        path, _ = _make_bench_tensor(n=10, seq_len=64)
        try:
            prov = BenchProxyProvider(path)
            out = prov.sample(device=torch.device("cpu"), k=4, seq_len=64)
            assert out.shape == (4, 64)
        finally:
            os.unlink(path)

    def test_truncates_to_seq_len(self):
        path, _ = _make_bench_tensor(n=10, seq_len=100)
        try:
            prov = BenchProxyProvider(path)
            out = prov.sample(device=torch.device("cpu"), k=3, seq_len=50)
            assert out.shape == (3, 50)
        finally:
            os.unlink(path)

    def test_rejects_short_sequences(self):
        """Shard with seq_len < requested must raise AssertionError — no padding allowed."""
        path, _ = _make_bench_tensor(n=10, seq_len=30)
        try:
            prov = BenchProxyProvider(path)
            raised = False
            try:
                prov.sample(device=torch.device("cpu"), k=3, seq_len=64)
            except AssertionError as e:
                raised = True
                assert "no padding" in str(e).lower()
            assert raised, "Should reject sequences shorter than seq_len"
        finally:
            os.unlink(path)

    def test_random_sampling_not_sequential(self):
        """Verify samples are drawn randomly, not in sequential order."""
        path, tokens = _make_bench_tensor(n=50, seq_len=32)
        try:
            prov = BenchProxyProvider(path)
            # Draw many batches and collect which rows we got
            seen_rows = set()
            for _ in range(20):
                out = prov.sample(device=torch.device("cpu"), k=4, seq_len=32)
                for row in out:
                    # find which original row matches
                    for i in range(tokens.size(0)):
                        if torch.equal(row, tokens[i]):
                            seen_rows.add(i)
                            break
            # With random sampling over 80 draws from 50 rows,
            # we should see significantly more than just the first few rows
            assert len(seen_rows) > 10, f"Only saw {len(seen_rows)} unique rows — looks sequential"
        finally:
            os.unlink(path)

    def test_samples_vary_across_calls(self):
        """Two consecutive calls should (almost certainly) return different samples."""
        path, _ = _make_bench_tensor(n=100, seq_len=32)
        try:
            prov = BenchProxyProvider(path)
            out1 = prov.sample(device=torch.device("cpu"), k=8, seq_len=32)
            out2 = prov.sample(device=torch.device("cpu"), k=8, seq_len=32)
            # With 100 rows and k=8, probability of identical draws is negligible
            assert not torch.equal(out1, out2), "Two random draws should differ"
        finally:
            os.unlink(path)

    def test_seen_counter_increments(self):
        path, _ = _make_bench_tensor(n=10, seq_len=32)
        try:
            prov = BenchProxyProvider(path)
            assert prov._seen == 0
            prov.sample(device=torch.device("cpu"), k=5, seq_len=32)
            assert prov._seen == 5
            prov.sample(device=torch.device("cpu"), k=3, seq_len=32)
            assert prov._seen == 8
        finally:
            os.unlink(path)

    def test_state_dict_roundtrip(self):
        path, _ = _make_bench_tensor(n=10, seq_len=32)
        try:
            prov = BenchProxyProvider(path)
            prov.sample(device=torch.device("cpu"), k=7, seq_len=32)
            state = prov.state_dict()
            assert state["seen"] == 7
            assert state["path"] == path

            prov2 = BenchProxyProvider(path)
            assert prov2._seen == 0
            prov2.load_state_dict(state)
            assert prov2._seen == 7
        finally:
            os.unlink(path)

    def test_rejects_non_2d_tensor(self):
        fd, path = tempfile.mkstemp(suffix=".pt")
        os.close(fd)
        torch.save(torch.randn(10), path)  # 1D tensor
        try:
            raised = False
            try:
                BenchProxyProvider(path)
            except ValueError:
                raised = True
            assert raised, "Should reject non-2D tensor"
        finally:
            os.unlink(path)

    def test_empty_shard_raises(self):
        fd, path = tempfile.mkstemp(suffix=".pt")
        os.close(fd)
        torch.save(torch.zeros(0, 32), path)  # empty
        try:
            prov = BenchProxyProvider(path)
            raised = False
            try:
                prov.sample(device=torch.device("cpu"), k=1, seq_len=32)
            except RuntimeError:
                raised = True
            assert raised, "Should raise on empty shard"
        finally:
            os.unlink(path)

    def test_k_larger_than_pool(self):
        """Requesting more samples than pool size should work (sampling with replacement)."""
        path, _ = _make_bench_tensor(n=5, seq_len=32)
        try:
            prov = BenchProxyProvider(path)
            out = prov.sample(device=torch.device("cpu"), k=20, seq_len=32)
            assert out.shape == (20, 32)
        finally:
            os.unlink(path)


# ===========================================================================
# RandomInDistributionProxyProvider tests
# ===========================================================================

class TestRandomInDistributionProxyProvider:

    def test_output_shape(self):
        loader = _make_loader(n=12, seq_len=32, batch_size=4)
        prov = RandomInDistributionProxyProvider(loader)
        out = prov.sample(device=torch.device("cpu"), k=3, seq_len=32)
        assert out.shape == (3, 32)

    def test_truncates_long_sequences(self):
        loader = _make_loader(n=8, seq_len=64, batch_size=4)
        prov = RandomInDistributionProxyProvider(loader)
        out = prov.sample(device=torch.device("cpu"), k=2, seq_len=30)
        assert out.shape == (2, 30)

    def test_rejects_short_sequences(self):
        """Loader with seq_len < requested must raise AssertionError — no padding allowed."""
        loader = _make_loader(n=8, seq_len=16, batch_size=4)
        prov = RandomInDistributionProxyProvider(loader)
        raised = False
        try:
            prov.sample(device=torch.device("cpu"), k=2, seq_len=64)
        except AssertionError as e:
            raised = True
            assert "no padding" in str(e).lower()
        assert raised, "Should reject sequences shorter than seq_len"

    def test_wraps_around_on_exhaustion(self):
        loader = _make_loader(n=4, seq_len=16, batch_size=4)
        prov = RandomInDistributionProxyProvider(loader)
        # First call exhausts the loader, second forces wrap-around
        prov.sample(device=torch.device("cpu"), k=4, seq_len=16)
        out = prov.sample(device=torch.device("cpu"), k=4, seq_len=16)
        assert out.shape == (4, 16)

    def test_seen_counter(self):
        loader = _make_loader(n=8, seq_len=16, batch_size=4)
        prov = RandomInDistributionProxyProvider(loader)
        prov.sample(device=torch.device("cpu"), k=3, seq_len=16)
        assert prov._seen == 3
        prov.sample(device=torch.device("cpu"), k=5, seq_len=16)
        assert prov._seen == 8

    def test_state_dict_roundtrip(self):
        loader = _make_loader(n=8, seq_len=16, batch_size=4)
        prov = RandomInDistributionProxyProvider(loader)
        prov.sample(device=torch.device("cpu"), k=6, seq_len=16)
        state = prov.state_dict()
        assert state["seen"] == 6

        prov2 = RandomInDistributionProxyProvider(loader)
        prov2.load_state_dict(state)
        assert prov2._seen == 6

    def test_k_larger_than_single_batch(self):
        """k > batch_size should pull from multiple batches."""
        loader = _make_loader(n=12, seq_len=16, batch_size=2)
        prov = RandomInDistributionProxyProvider(loader)
        out = prov.sample(device=torch.device("cpu"), k=7, seq_len=16)
        assert out.shape == (7, 16)
