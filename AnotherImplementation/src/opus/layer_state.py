from dataclasses import dataclass

import torch
from torch import Tensor

from .countsketch import TensorSketchOperator
from .preconditioner import compute_factored_preconditioner


@dataclass
class LayerConfig:
    """Static configuration for a single registered layer."""

    name: str
    d_in: int
    d_out: int


class LayerSketchState:
    """
    Manages all sketch-related state for a single layer in one training step.

    Owns:
        - TensorSketchOperator: precomputed hash/sign arrays for this layer's dims
        - proxy_sketch ψ^(t,r):  (m,) averaged sketch of proxy samples (unpreconditioned)
        - history Φ^(t,r):       (m,) accumulated sketch of already-selected candidates

    Proxy sketches are computed WITHOUT the preconditioner (raw gradient direction).
    Candidate sketches are computed WITH the factored preconditioner applied.

    Both proxy and candidate (a, b) pairs should be passed in and can be
    discarded immediately after calling the corresponding method — this class
    retains only the small sketch vectors.
    """

    def __init__(
        self,
        config: LayerConfig,
        sketch_dim: int,
        device: torch.device,
        seed: int = 42,
    ):
        self.config = config
        self.sketch_dim = sketch_dim
        self.device = device

        self.sketch_op = TensorSketchOperator(
            d_in=config.d_in,
            d_out=config.d_out,
            sketch_dim=sketch_dim,
            device=device,
            seed=seed,
        )

        self.proxy_sketch: Tensor | None = None
        self.history: Tensor = torch.zeros(sketch_dim, device=device)

    def compute_proxy_sketch(self, a: Tensor, b: Tensor) -> None:
        """
        Sketch the proxy batch and store the averaged result as ψ^(t,r).
        No preconditioner is applied — proxy represents the raw gradient direction.

        Args:
            a: (K_proxy, d_in)  proxy input activations
            b: (K_proxy, d_out) proxy error signals
        """
        sketches = self.sketch_op.sketch(a, b)  # (K_proxy, m)
        self.proxy_sketch = sketches.mean(dim=0)  # (m,)

    def compute_candidate_sketches(
        self,
        a: Tensor,
        b: Tensor,
        v: Tensor,
        eps: float = 1e-8,
    ) -> Tensor:
        """
        Compute preconditioned sketches for all candidates in this layer.
        The factored preconditioner is applied before sketching.

        Args:
            a: (N, d_in)   candidate input activations
            b: (N, d_out)  candidate error signals
            v: (d_out, d_in) AdamW second moment buffer for this layer
            eps: stability constant for preconditioner

        Returns:
            (N, m) preconditioned sketch per candidate
        """
        u, w = compute_factored_preconditioner(v, eps=eps)

        # Pre-scale: absorb separable preconditioner into factors
        b_tilde = b * u.unsqueeze(0)  # (N, d_out)
        a_tilde = a * w.unsqueeze(0)  # (N, d_in)

        return self.sketch_op.sketch(a_tilde, b_tilde)  # (N, m)

    def update_history(self, candidate_sketch: Tensor) -> None:
        """
        Add a selected candidate's sketch to the running history Φ^(t,r).
        Called once per selected candidate, before scoring the next one.

        Args:
            candidate_sketch: (m,) sketch of the selected candidate for this layer
        """
        self.history.add_(candidate_sketch)

    def reset(self) -> None:
        """Clear per-step state. Call at the start of each new training step."""
        self.history.zero_()
        self.proxy_sketch = None
