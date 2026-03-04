import torch

from production.opus.countsketch import CountSketchProjector


def test_countsketch_matches_naive_small_case():
    torch.manual_seed(0)
    m = 257
    proj = CountSketchProjector(sketch_dim=m, seed=11, row_chunk_size=2)

    T, in_dim, out_dim = 5, 4, 3
    a = torch.randn(T, in_dim)
    g = torch.randn(T, out_dim)
    p = torch.randn(out_dim, in_dim).abs() + 0.1

    fast = proj.project_linear_sample(a, g, p, out_dim=out_dim, in_dim=in_dim)

    cache = proj._get_cache(out_dim, in_dim, a.device)
    grad = g.transpose(0, 1).matmul(a) * p
    naive = torch.zeros(m)
    for o in range(out_dim):
        for i in range(in_dim):
            h = int((cache.row_hash[o] + cache.col_hash[i]).item() % m)
            s = float((cache.row_sign[o] * cache.col_sign[i]).item())
            naive[h] += s * float(grad[o, i].item())

    assert torch.allclose(fast.cpu(), naive, atol=1e-5, rtol=1e-5)


def test_countsketch_batch_shape():
    torch.manual_seed(1)
    proj = CountSketchProjector(sketch_dim=128, seed=7)
    a = torch.randn(6, 3, 8)
    g = torch.randn(6, 3, 5)
    out = proj.project_linear_batch(a, g, None, out_dim=5, in_dim=8)
    assert out.shape == (6, 128)


def test_countsketch_sanitizes_nonfinite_inputs():
    proj = CountSketchProjector(sketch_dim=64, seed=3)
    a = torch.randn(2, 4)
    g = torch.randn(2, 3)
    p = torch.ones(3, 4)
    a[0, 0] = torch.nan
    g[1, 1] = torch.inf
    p[2, 2] = -torch.inf

    out = proj.project_linear_sample(a, g, p, out_dim=3, in_dim=4)
    assert out.shape == (64,)
    assert bool(torch.isfinite(out).all())


def test_countsketch_pair_projection_matches_two_single_calls():
    torch.manual_seed(3)
    proj = CountSketchProjector(sketch_dim=128, seed=9, row_chunk_size=4)
    left = torch.randn(7, 5)
    right_a = torch.randn(7, 6)
    right_b = torch.randn(7, 6)
    p_a = torch.randn(5, 6).abs() + 0.1
    p_b = torch.randn(5, 6).abs() + 0.1

    single_a = proj.project_outer_sum_sample(
        left=left,
        right=right_a,
        preconditioner=p_a,
        row_dim=5,
        col_dim=6,
        out_dtype=torch.float32,
    )
    single_b = proj.project_outer_sum_sample(
        left=left,
        right=right_b,
        preconditioner=p_b,
        row_dim=5,
        col_dim=6,
        out_dtype=torch.float32,
    )
    pair_a, pair_b = proj.project_outer_sum_pair_sample(
        left=left,
        right_a=right_a,
        right_b=right_b,
        preconditioner_a=p_a,
        preconditioner_b=p_b,
        row_dim=5,
        col_dim=6,
        out_dtype=torch.float32,
    )

    assert torch.allclose(single_a, pair_a, atol=1e-5, rtol=1e-5)
    assert torch.allclose(single_b, pair_b, atol=1e-5, rtol=1e-5)


def test_countsketch_is_param_key_specific():
    torch.manual_seed(4)
    proj = CountSketchProjector(sketch_dim=256, seed=17)
    a = torch.randn(6, 7)
    g = torch.randn(6, 5)
    p = torch.ones(5, 7)

    s_a = proj.project_linear_sample(
        a,
        g,
        p,
        out_dim=5,
        in_dim=7,
        sketch_key="layerA.weight",
    )
    s_b = proj.project_linear_sample(
        a,
        g,
        p,
        out_dim=5,
        in_dim=7,
        sketch_key="layerB.weight",
    )
    s_a_repeat = proj.project_linear_sample(
        a,
        g,
        p,
        out_dim=5,
        in_dim=7,
        sketch_key="layerA.weight",
    )

    assert torch.allclose(s_a, s_a_repeat, atol=1e-6, rtol=1e-6)
    assert not torch.allclose(s_a, s_b, atol=1e-6, rtol=1e-6)
