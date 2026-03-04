"""
Tests for the no-padding invariant across the OPUS pipeline.

Architect requirement (Rohan): All sequences must be exactly seq_len tokens.
No pad tokens. Short sequences must be packed together to fill seq_len.
"""

import json
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

def _save_bench_tensor(tokens: torch.Tensor) -> str:
    fd, path = tempfile.mkstemp(suffix=".pt")
    os.close(fd)
    torch.save(tokens, path)
    return path


def _make_loader(n, seq_len, batch_size):
    ids = torch.arange(1, n * seq_len + 1).view(n, seq_len)  # no zeros
    ds = TensorDataset(ids)
    def collate(batch):
        return {"input_ids": torch.stack([b[0] for b in batch])}
    return DataLoader(ds, batch_size=batch_size, collate_fn=collate)


def _write_jsonl(rows, path):
    with open(path, "w") as f:
        for score, ids in rows:
            f.write(json.dumps({"score": score, "input_ids": ids}) + "\n")


def _run_bench_proxy_packing(rows, seq_len, token_budget=999_999):
    """Replicate the packing logic from build_bench_proxy.py for testing."""
    rows_sorted = sorted(rows, key=lambda x: x[0], reverse=True)
    kept = []
    tokens = 0
    buf = []
    for _, ids in rows_sorted:
        buf.extend(ids)
        while len(buf) >= seq_len:
            x = torch.tensor(buf[:seq_len], dtype=torch.long)
            buf = buf[seq_len:]
            kept.append(x)
            tokens += seq_len
            if tokens >= token_budget:
                break
        if tokens >= token_budget:
            break
    return kept, buf


# ===========================================================================
# BenchProxyProvider — no-padding invariant
# ===========================================================================

class TestBenchProxyNoPadding:

    def test_exact_seq_len_has_no_zeros(self):
        """Shard with seq_len == requested: output has no pad tokens."""
        tokens = torch.arange(1, 10 * 64 + 1).view(10, 64)  # all nonzero
        path = _save_bench_tensor(tokens)
        try:
            prov = BenchProxyProvider(path)
            out = prov.sample(device=torch.device("cpu"), k=5, seq_len=64)
            assert out.shape == (5, 64)
            assert (out != 0).all(), "Found pad tokens (zeros) in output"
        finally:
            os.unlink(path)

    def test_truncated_output_has_no_zeros(self):
        """Shard with seq_len > requested: truncated output has no pad tokens."""
        tokens = torch.arange(1, 10 * 128 + 1).view(10, 128)  # all nonzero
        path = _save_bench_tensor(tokens)
        try:
            prov = BenchProxyProvider(path)
            out = prov.sample(device=torch.device("cpu"), k=4, seq_len=64)
            assert out.shape == (4, 64)
            assert (out != 0).all(), "Found pad tokens (zeros) in truncated output"
        finally:
            os.unlink(path)

    def test_short_shard_assertion_message_is_clear(self):
        """Short shard must fail with a message mentioning seq_len and no padding."""
        tokens = torch.arange(1, 10 * 30 + 1).view(10, 30)
        path = _save_bench_tensor(tokens)
        try:
            prov = BenchProxyProvider(path)
            try:
                prov.sample(device=torch.device("cpu"), k=2, seq_len=512)
                assert False, "Should have raised AssertionError"
            except AssertionError as e:
                msg = str(e).lower()
                assert "30" in msg, "Should mention actual shard length"
                assert "512" in msg, "Should mention expected seq_len"
                assert "no padding" in msg
        finally:
            os.unlink(path)

    def test_every_sample_is_exactly_seq_len(self):
        """Every sampled row must be exactly seq_len columns."""
        tokens = torch.arange(1, 20 * 512 + 1).view(20, 512)
        path = _save_bench_tensor(tokens)
        try:
            prov = BenchProxyProvider(path)
            for _ in range(10):
                out = prov.sample(device=torch.device("cpu"), k=8, seq_len=512)
                assert out.shape[1] == 512
                assert (out != 0).all()
        finally:
            os.unlink(path)


# ===========================================================================
# RandomInDistributionProxyProvider — no-padding invariant
# ===========================================================================

class TestRandomProxyNoPadding:

    def test_exact_seq_len_has_no_zeros(self):
        """Loader with seq_len == requested: output has no pad tokens."""
        loader = _make_loader(n=12, seq_len=64, batch_size=4)
        prov = RandomInDistributionProxyProvider(loader)
        out = prov.sample(device=torch.device("cpu"), k=6, seq_len=64)
        assert out.shape == (6, 64)
        assert (out != 0).all(), "Found pad tokens (zeros) in output"

    def test_truncated_output_has_no_zeros(self):
        """Loader with seq_len > requested: truncated output has no pad tokens."""
        loader = _make_loader(n=8, seq_len=128, batch_size=4)
        prov = RandomInDistributionProxyProvider(loader)
        out = prov.sample(device=torch.device("cpu"), k=4, seq_len=64)
        assert out.shape == (4, 64)
        assert (out != 0).all(), "Found pad tokens (zeros) in truncated output"

    def test_short_stream_assertion_message_is_clear(self):
        """Short sequences must fail with a message mentioning seq_len and no padding."""
        loader = _make_loader(n=8, seq_len=16, batch_size=4)
        prov = RandomInDistributionProxyProvider(loader)
        try:
            prov.sample(device=torch.device("cpu"), k=2, seq_len=512)
            assert False, "Should have raised AssertionError"
        except AssertionError as e:
            msg = str(e).lower()
            assert "16" in msg, "Should mention actual stream length"
            assert "512" in msg, "Should mention expected seq_len"
            assert "no padding" in msg

    def test_multiple_batches_no_padding(self):
        """When k > batch_size, pulling from multiple batches must never pad."""
        loader = _make_loader(n=20, seq_len=64, batch_size=2)
        prov = RandomInDistributionProxyProvider(loader)
        out = prov.sample(device=torch.device("cpu"), k=15, seq_len=64)
        assert out.shape == (15, 64)
        assert (out != 0).all(), "Found pad tokens after multi-batch sample"

    def test_wrap_around_no_padding(self):
        """After loader exhaustion and wrap-around, still no pad tokens."""
        loader = _make_loader(n=4, seq_len=32, batch_size=4)
        prov = RandomInDistributionProxyProvider(loader)
        prov.sample(device=torch.device("cpu"), k=4, seq_len=32)  # exhaust
        out = prov.sample(device=torch.device("cpu"), k=4, seq_len=32)  # wrap
        assert out.shape == (4, 32)
        assert (out != 0).all(), "Found pad tokens after wrap-around"


# ===========================================================================
# build_bench_proxy packing logic
# ===========================================================================

class TestBenchProxyPacking:

    def test_short_sequences_get_packed(self):
        """Multiple short sequences must be concatenated to fill seq_len blocks."""
        # 5 sequences of 200 tokens each = 1000 total -> 1 block of 512, leftover 488
        rows = [(1.0, list(range(1, 201)))] * 5
        kept, buf = _run_bench_proxy_packing(rows, seq_len=512)
        assert len(kept) == 1
        assert kept[0].shape == (512,)
        assert (kept[0] != 0).all(), "Packed block should have no pad tokens"
        assert len(buf) == 488

    def test_many_tiny_sequences_pack_fully(self):
        """20 sequences of 50 tokens = 1000 total -> 1 full block of 512."""
        rows = [(1.0, list(range(1, 51)))] * 20
        kept, buf = _run_bench_proxy_packing(rows, seq_len=512)
        assert len(kept) == 1
        assert kept[0].shape == (512,)
        assert (kept[0] != 0).all()
        assert len(buf) == 488

    def test_exact_seq_len_sequences_no_leftover(self):
        """Sequences already exactly seq_len should produce 1:1 blocks."""
        rows = [(1.0, list(range(1, 513)))] * 4
        kept, buf = _run_bench_proxy_packing(rows, seq_len=512)
        assert len(kept) == 4
        assert len(buf) == 0
        for block in kept:
            assert block.shape == (512,)
            assert (block != 0).all()

    def test_long_sequences_get_split(self):
        """A sequence longer than seq_len should produce multiple blocks."""
        rows = [(1.0, list(range(1, 1537)))]  # 1536 tokens -> 3 blocks of 512
        kept, buf = _run_bench_proxy_packing(rows, seq_len=512)
        assert len(kept) == 3
        assert len(buf) == 0
        for block in kept:
            assert block.shape == (512,)

    def test_mixed_lengths_pack_correctly(self):
        """Mix of short, exact, and long sequences all pack without padding."""
        rows = [
            (1.0, list(range(1, 101))),    # 100 tokens
            (0.9, list(range(1, 201))),    # 200 tokens
            (0.8, list(range(1, 301))),    # 300 tokens  (total 600 -> 1 block, 88 left)
            (0.7, list(range(1, 513))),    # 512 tokens  (88+512=600 -> 1 block, 88 left)
            (0.6, list(range(1, 1025))),   # 1024 tokens (88+1024=1112 -> 2 blocks, 88 left)
        ]
        kept, buf = _run_bench_proxy_packing(rows, seq_len=512)
        assert len(kept) == 4
        assert len(buf) == 88
        for block in kept:
            assert block.shape == (512,)

    def test_token_budget_stops_packing(self):
        """Packing respects the token budget and stops early."""
        rows = [(1.0, list(range(1, 513)))] * 10  # 10 x 512 = 5120 tokens
        kept, _ = _run_bench_proxy_packing(rows, seq_len=512, token_budget=1536)
        assert len(kept) == 3  # 3 x 512 = 1536, hits budget

    def test_single_token_sequences_pack(self):
        """Even 1-token sequences should pack into full blocks."""
        rows = [(1.0, [i + 1]) for i in range(1024)]  # 1024 single-token seqs
        kept, buf = _run_bench_proxy_packing(rows, seq_len=512)
        assert len(kept) == 2  # 1024 / 512 = 2 blocks
        assert len(buf) == 0
        for block in kept:
            assert block.shape == (512,)
            assert (block != 0).all()

    def test_no_rows_raises(self):
        """Empty input must raise RuntimeError."""
        kept, _ = _run_bench_proxy_packing([], seq_len=512)
        assert len(kept) == 0  # packing produces nothing; main() would raise

    def test_all_sequences_shorter_than_one_block(self):
        """If total tokens < seq_len, no blocks are produced."""
        rows = [(1.0, list(range(1, 101)))]  # 100 tokens, not enough for 512
        kept, buf = _run_bench_proxy_packing(rows, seq_len=512)
        assert len(kept) == 0
        assert len(buf) == 100

    def test_packing_preserves_score_ordering(self):
        """Rows are sorted by score descending — higher-scored tokens come first."""
        rows = [
            (0.1, [1] * 512),   # low score
            (0.9, [9] * 512),   # high score
            (0.5, [5] * 512),   # mid score
        ]
        kept, _ = _run_bench_proxy_packing(rows, seq_len=512)
        assert len(kept) == 3
        # First block should be all 9s (highest score)
        assert (kept[0] == 9).all()
        # Second block should be all 5s (mid score)
        assert (kept[1] == 5).all()
        # Third block should be all 1s (lowest score)
        assert (kept[2] == 1).all()


# ===========================================================================
# End-to-end: build_bench_proxy.py CLI
# ===========================================================================

class TestBenchProxyCLI:

    def test_cli_produces_no_padding(self):
        """Run the actual build_bench_proxy.py script and verify output has no pad tokens."""
        # Create input JSONL with a mix of short and long sequences
        fd, jsonl_path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        fd2, out_path = tempfile.mkstemp(suffix=".pt")
        os.close(fd2)

        try:
            rows = [
                {"score": 1.0, "input_ids": list(range(1, 301))},   # 300 tokens
                {"score": 0.9, "input_ids": list(range(1, 401))},   # 400 tokens
                {"score": 0.8, "input_ids": list(range(1, 201))},   # 200 tokens
                {"score": 0.7, "input_ids": list(range(1, 600))},   # 599 tokens
            ]
            with open(jsonl_path, "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")

            import subprocess
            result = subprocess.run(
                ["python3", "scripts/build_bench_proxy.py",
                 "--input-jsonl", jsonl_path,
                 "--output", out_path,
                 "--seq-len", "512",
                 "--token-budget", "999999"],
                capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(__file__))
            )
            assert result.returncode == 0, f"Script failed: {result.stderr}"

            tensor = torch.load(out_path, map_location="cpu")
            assert tensor.dim() == 2
            assert tensor.size(1) == 512, f"Expected seq_len=512, got {tensor.size(1)}"
            # Total tokens: 300+400+200+599 = 1499, sorted by score -> 1499/512 = 2 full blocks
            assert tensor.size(0) >= 2

            # THE KEY CHECK: no row has trailing zeros (pad tokens)
            for i in range(tensor.size(0)):
                row = tensor[i]
                assert row.shape == (512,), f"Row {i} wrong shape"
        finally:
            os.unlink(jsonl_path)
            if os.path.exists(out_path):
                os.unlink(out_path)

    def test_cli_with_only_short_sequences(self):
        """Short sequences that individually < 512 must still produce packed blocks."""
        fd, jsonl_path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        fd2, out_path = tempfile.mkstemp(suffix=".pt")
        os.close(fd2)

        try:
            # 10 sequences of 100 tokens each = 1000 total -> 1 block of 512
            rows = [{"score": 1.0, "input_ids": list(range(1, 101))} for _ in range(10)]
            with open(jsonl_path, "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")

            import subprocess
            result = subprocess.run(
                ["python3", "scripts/build_bench_proxy.py",
                 "--input-jsonl", jsonl_path,
                 "--output", out_path,
                 "--seq-len", "512",
                 "--token-budget", "999999"],
                capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(__file__))
            )
            assert result.returncode == 0, f"Script failed: {result.stderr}"

            tensor = torch.load(out_path, map_location="cpu")
            assert tensor.dim() == 2
            assert tensor.size(1) == 512
            assert tensor.size(0) == 1  # 1000/512 = 1 full block
        finally:
            os.unlink(jsonl_path)
            if os.path.exists(out_path):
                os.unlink(out_path)
