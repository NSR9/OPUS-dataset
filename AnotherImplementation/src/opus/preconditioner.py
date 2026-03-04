import torch
from torch import Tensor


def compute_factored_preconditioner(
    v: Tensor,
    eps: float = 1e-8,
) -> tuple[Tensor, Tensor]:
    """
    Compute an Adafactor-style factored approximation of AdamW's diagonal
    preconditioner, making it separable so TensorSketch can operate in
    O(d_out + d_in) instead of O(d_out * d_in).

    Problem:
        AdamW's true preconditioner is p_ij = 1/sqrt(v_ij + eps), where v has
        shape (d_out, d_in). This is not separable — you cannot write
        p_ij = u_i * w_j in general, so you cannot absorb it into b and a
        separately without iterating over the full outer product.

    Solution:
        Approximate the second moment matrix as a rank-1 outer product
        (Adafactor-style):
            v_ij ≈ (v_row[i] * v_col[j]) / mean(v)

        This gives a separable preconditioner:
            p_ij ≈ u_i * w_j

        where (dropping the shared scalar mean(v)^0.5 since it doesn't affect
        candidate ranking):
            u_i = 1 / sqrt(v_row[i] + eps)   shape: (d_out,)
            w_j = 1 / sqrt(v_col[j] + eps)   shape: (d_in,)

        Pre-scaling:
            b̃ = u ⊙ b
            ã = w ⊙ a

        Then TensorSketch(b̃ ⊗ ã) ≈ Sketch of preconditioned gradient.

    Note:
        This approximation is used ONLY for OPUS scoring. AdamW training
        updates are completely unaffected.

    Args:
        v:   (d_out, d_in) second moment buffer from AdamW optimizer state
        eps: numerical stability constant (should match AdamW's eps)

    Returns:
        u: (d_out,) row-wise preconditioner scale factors
        w: (d_in,)  col-wise preconditioner scale factors
    """
    v_row = v.mean(dim=1)  # (d_out,): mean over input dim
    v_col = v.mean(dim=0)  # (d_in,):  mean over output dim

    u = 1.0 / torch.sqrt(v_row + eps)  # (d_out,)
    w = 1.0 / torch.sqrt(v_col + eps)  # (d_in,)

    return u, w
