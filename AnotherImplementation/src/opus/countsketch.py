import torch
from torch import Tensor


class TensorSketchOperator:
    """
    Implements TensorSketch for efficiently sketching outer products b ⊗ a
    without materializing the full (d_out × d_in) matrix.

    Core idea: maintain two independent CountSketch operators, one per dimension.
    The sketch of the outer product is then the circular convolution of the two
    individual sketches, computed efficiently via FFT:

        Sketch(b ⊗ a) = IFFT(FFT(CS_b(b)) * FFT(CS_a(a))).real

    This preserves inner products of outer products in expectation:
        E[<Sketch(b1 ⊗ a1), Sketch(b2 ⊗ a2)>] = <b1 ⊗ a1, b2 ⊗ a2>
                                                 = <b1, b2> * <a1, a2>

    Supports both single vectors (d,) and batched inputs (N, d).
    Hash and sign arrays are precomputed once and reused across all steps.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        sketch_dim: int,
        device: torch.device,
        seed: int = 42,
    ):
        self.d_in = d_in
        self.d_out = d_out
        self.sketch_dim = sketch_dim
        self.device = device

        # Use a CPU generator for reproducibility, then move arrays to device
        rng = torch.Generator(device="cpu")
        rng.manual_seed(seed)

        # Hash arrays: map each coordinate index to a sketch bucket in [0, m)
        self.hash_a = torch.randint(0, sketch_dim, (d_in,), generator=rng).to(device)
        self.hash_b = torch.randint(0, sketch_dim, (d_out,), generator=rng).to(device)

        # Sign arrays: random ±1 for variance reduction
        self.sign_a = (
            torch.randint(0, 2, (d_in,), generator=rng).to(device) * 2 - 1
        ).float()
        self.sign_b = (
            torch.randint(0, 2, (d_out,), generator=rng).to(device) * 2 - 1
        ).float()

    def _count_sketch_1d(self, x: Tensor, hash_arr: Tensor, sign_arr: Tensor) -> Tensor:
        """
        Apply CountSketch to a vector or batch of vectors.

        For each coordinate j: sketch[hash[j]] += sign[j] * x[j]

        Args:
            x:        (d,) or (N, d)
            hash_arr: (d,) bucket indices in [0, m)
            sign_arr: (d,) ±1 signs

        Returns:
            (m,) or (N, m) sketch
        """
        signed = x * sign_arr  # broadcast over batch dim if present

        if x.dim() == 1:
            sketch = torch.zeros(self.sketch_dim, device=x.device, dtype=x.dtype)
            sketch.index_add_(0, hash_arr, signed)
        else:
            N = x.shape[0]
            sketch = torch.zeros(N, self.sketch_dim, device=x.device, dtype=x.dtype)
            # hash_arr is shared across all samples in the batch
            hash_expanded = hash_arr.unsqueeze(0).expand(N, -1)  # (N, d)
            sketch.scatter_add_(1, hash_expanded, signed)

        return sketch

    def sketch(self, a: Tensor, b: Tensor) -> Tensor:
        """
        Compute TensorSketch of outer product b ⊗ a.

        Args:
            a: (d_in,) or (N, d_in)  — input activations (possibly pre-scaled)
            b: (d_out,) or (N, d_out) — error signals   (possibly pre-scaled)

        Returns:
            (m,) or (N, m) sketch of the outer product
        """
        cs_a = self._count_sketch_1d(a, self.hash_a, self.sign_a)
        cs_b = self._count_sketch_1d(b, self.hash_b, self.sign_b)

        # Circular convolution via FFT = TensorSketch of outer product
        result = torch.fft.ifft(
            torch.fft.fft(cs_b, dim=-1) * torch.fft.fft(cs_a, dim=-1),
            dim=-1,
        ).real

        return result
