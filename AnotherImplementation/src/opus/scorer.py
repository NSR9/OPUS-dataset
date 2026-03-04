import torch
import torch.nn.functional as F
from torch import Tensor

from .layer_state import LayerConfig, LayerSketchState


class OPUSScorer:
    """
    Top-level OPUS scoring module.

    Orchestrates the full per-step pipeline:

        1. update_proxy_sketch()      — called per layer after proxy forward-backward
        2. update_candidate_sketches() — called per layer after candidate forward-backward
           (a, b tensors can be discarded immediately after each call)
        3. run_selection(K)           — iteratively selects K candidates via:
               a. compute_utilities() — alignment minus redundancy, summed over layers
               b. boltzmann_select()  — softmax sampling with temperature τ
               c. update_history()    — update Φ with selected candidate's sketches

    The caller is responsible for supplying (a, b) pairs per layer (e.g. via hooks)
    and the corresponding AdamW v buffers from the optimizer state dict.

    State owned:
        - layer_states:        Dict[str, LayerSketchState]  — per-layer sketch state
        - _candidate_sketches: Dict[str, Tensor]            — (N, m) per layer, current step
    """

    def __init__(
        self,
        sketch_dim: int = 8192,
        temperature: float = 0.9,
        eta: float = 1.0,
        eps: float = 1e-8,
        device: torch.device | None = None,
    ):
        """
        Args:
            sketch_dim:  Dimension m of CountSketch projection vectors
            temperature: Boltzmann sampling temperature τ
            eta:         Learning rate scalar η used in utility formula
            eps:         Stability constant passed to factored preconditioner
            device:      Target device for all tensors
        """
        self.sketch_dim = sketch_dim
        self.temperature = temperature
        self.eta = eta
        self.eps = eps
        self.device = device or torch.device("cpu")

        self.layer_states: dict[str, LayerSketchState] = {}
        self._candidate_sketches: dict[str, Tensor] = {}
        self._n_candidates: int | None = None

    def register_layer(
        self,
        name: str,
        d_in: int,
        d_out: int,
        global_seed: int = 42,
    ) -> None:
        """
        Register a layer to participate in scoring.
        Must be called before any step begins.

        Args:
            name:  Unique identifier for the layer (e.g. 'blocks.0.attn.qkv_proj')
            d_in:  Input dimension of the weight matrix
            d_out: Output dimension of the weight matrix
            seed:  RNG seed for this layer's CountSketch hash/sign arrays.
                   Use different seeds per layer to ensure independence.
        """
        # Derive a deterministic per-layer seed from the global seed and layer name.
        # This ensures:
        #   1. Reproducibility — same global seed + same layer name = same sketch operator
        #   2. Independence between layers — different names = different hash/sign arrays
        layer_seed = global_seed ^ hash(name) & 0xFFFFFFFF
        config = LayerConfig(name=name, d_in=d_in, d_out=d_out)
        self.layer_states[name] = LayerSketchState(
            config=config,
            sketch_dim=self.sketch_dim,
            device=self.device,
            seed=layer_seed,
        )

    def update_proxy_sketch(
        self,
        layer_name: str,
        a: Tensor,
        b: Tensor,
    ) -> None:
        """
        Compute and store the proxy sketch ψ^(t,r) for a layer.
        No preconditioner is applied — proxy direction is raw gradient space.
        a and b may be freed after this call.

        Args:
            layer_name: registered layer name
            a: (K_proxy, d_in)  proxy input activations
            b: (K_proxy, d_out) proxy error signals
        """
        self._get_layer(layer_name).compute_proxy_sketch(a, b)

    def update_candidate_sketches(
        self,
        layer_name: str,
        a: Tensor,
        b: Tensor,
        v: Tensor,
    ) -> None:
        """
        Compute and store preconditioned candidate sketches ϕ^(t,r)(z) for a layer.
        a and b may be freed after this call — only the (N, m) sketches are retained.

        Args:
            layer_name: registered layer name
            a: (N, d_in)     candidate input activations
            b: (N, d_out)    candidate error signals
            v: (d_out, d_in) AdamW second moment buffer for this layer
        """
        sketches = self._get_layer(layer_name).compute_candidate_sketches(
            a, b, v, eps=self.eps
        )
        self._candidate_sketches[layer_name] = sketches

        if self._n_candidates is None:
            self._n_candidates = sketches.shape[0]
        else:
            assert sketches.shape[0] == self._n_candidates, (
                f"Candidate count mismatch for layer '{layer_name}': "
                f"expected {self._n_candidates}, got {sketches.shape[0]}"
            )

    def run_selection(self, k: int) -> list[int]:
        assert self._n_candidates is not None
        assert k <= self._n_candidates

        # Validate all proxy sketches are set
        for name, state in self.layer_states.items():
            assert state.proxy_sketch is not None, (
                f"Proxy sketch not computed for layer '{name}'. "
                "Call update_proxy_sketch() before run_selection()."
            )

        selected: list[int] = []
        remaining = torch.ones(self._n_candidates, dtype=torch.bool, device=self.device)

        # Precompute alignment once — proxy sketch is fixed for the whole step
        alignment = torch.zeros(self._n_candidates, device=self.device)
        for name, state in self.layer_states.items():
            phi = self._candidate_sketches[name]  # (N, m)
            alignment += phi @ state.proxy_sketch  # type: ignore[operator]
        alignment *= self.eta

        for _ in range(k):
            utilities = self._compute_utilities(alignment)
            utilities[~remaining] = float("-inf")

            idx = self._boltzmann_select(utilities)
            selected.append(idx)
            remaining[idx] = False
            self._update_history(idx)

        return selected

    def reset_step(self) -> None:
        """
        Clear all per-step state. Call at the start of each new training step
        before processing proxy and candidate passes.
        """
        self._candidate_sketches.clear()
        self._n_candidates = None
        for state in self.layer_states.values():
            state.reset()

    def _compute_utilities(self, alignment: Tensor) -> Tensor:
        """
        Compute utility scores using precomputed alignment and current history Φ.
        Not intended for external use — call run_selection() instead.

        Args:
            alignment: (N,) precomputed alignment term η * Σ_r <ϕ(z), ψ>

        Returns:
            (N,) utility scores
        """
        redundancy = torch.zeros(self._n_candidates, device=self.device)  # type: ignore[arg-type]
        for name, state in self.layer_states.items():
            phi = self._candidate_sketches[name]  # (N, m)
            redundancy += phi @ state.history  # (N,)
        redundancy *= self.eta**2

        return alignment - redundancy

    def _boltzmann_select(self, utilities: Tensor) -> int:
        """
        Sample one candidate index proportionally to exp(U / τ).
        Candidates with utility = -inf (already selected) get probability 0.

        Args:
            utilities: (N,) utility scores, with -inf masking selected candidates

        Returns:
            Sampled candidate index
        """
        probs = F.softmax(utilities / self.temperature, dim=0)
        return int(torch.multinomial(probs, num_samples=1).item())

    def _update_history(self, selected_idx: int) -> None:
        """
        Add the selected candidate's sketch to Φ^(t,r) for all layers.
        Must be called after each selection, before computing utilities again.

        Args:
            selected_idx: index of the selected candidate
        """
        for name, state in self.layer_states.items():
            state.update_history(self._candidate_sketches[name][selected_idx])

    def _get_layer(self, name: str) -> LayerSketchState:
        assert name in self.layer_states, (
            f"Layer '{name}' not registered. Call register_layer() first."
        )
        return self.layer_states[name]
