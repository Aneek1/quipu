import pytest
import torch

from quipu.attnres import BlockAttnRes

D = 16


def _steps(n_layer: int, seed: int = 0):
    """A deterministic, per-token, nonlinear stand-in for layer l."""
    g = torch.Generator().manual_seed(seed)
    mats = [torch.randn(D, D, generator=g) / D ** 0.5 for _ in range(n_layer)]
    return lambda l, h: torch.tanh(h @ mats[l])


def _reference_sources(l: int, x0, outs, layers_per_block):
    """Kimi K3 source list for layer l, written out by block index, independent of
    BlockAttnRes: b_0 = embedding; completed blocks are sums of their layer outputs;
    after the first layer of a block, the block's partial sum is appended."""
    n, i = divmod(l, layers_per_block)          # block n (0-based), position i in it
    srcs = [x0]
    for b in range(n):
        srcs.append(sum(outs[b * layers_per_block:(b + 1) * layers_per_block]))
    if i > 0:
        srcs.append(sum(outs[n * layers_per_block:l]))
    return srcs


def _reference_forward(ar: BlockAttnRes, x0, step):
    """AttnRes by the reference source lists; returns (inputs, outputs, final)."""
    lpb = ar.layers_per_block
    inputs, outs = [], []
    for l in range(ar.n_layer):
        srcs = _reference_sources(l, x0, outs, lpb)
        v = torch.stack(srcs)
        k = torch.nn.functional.rms_norm(v, (D,), eps=ar.eps)
        alpha = (k @ ar.queries[l]).softmax(0)
        h = (alpha[..., None] * v).sum(0)
        inputs.append(h)
        outs.append(step(l, h))
    final_srcs = [x0] + [sum(outs[b * lpb:(b + 1) * lpb]) for b in range(ar.n_blocks)]
    v = torch.stack(final_srcs)
    k = torch.nn.functional.rms_norm(v, (D,), eps=ar.eps)
    alpha = (k @ ar.queries[ar.n_layer]).softmax(0)
    return inputs, outs, (alpha[..., None] * v).sum(0)


def _recording(step):
    inputs, outs = [], []

    def rec(l, h):
        inputs.append(h)
        o = step(l, h)
        outs.append(o)
        return o
    return rec, inputs, outs


def _randomise(ar: BlockAttnRes, seed: int = 1) -> None:
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for q in ar.queries:
            q.copy_(torch.randn(D, generator=g))


def test_pseudo_queries_start_at_zero_one_per_layer_plus_the_head():
    ar = BlockAttnRes(D, n_layer=6, n_blocks=3)
    assert len(ar.queries) == 7
    assert all(q.shape == (D,) and torch.count_nonzero(q) == 0 for q in ar.queries)


@pytest.mark.parametrize("n_layer, n_blocks", [(4, 3), (0, 1), (4, 0)])
def test_rejects_layers_that_do_not_split_into_blocks(n_layer, n_blocks):
    with pytest.raises(ValueError):
        BlockAttnRes(D, n_layer=n_layer, n_blocks=n_blocks)


@pytest.mark.parametrize("n_layer, n_blocks", [(6, 3), (6, 2), (4, 4), (4, 1)])
def test_matches_the_reference_source_lists_with_learned_queries(n_layer, n_blocks):
    torch.manual_seed(0)
    ar = BlockAttnRes(D, n_layer=n_layer, n_blocks=n_blocks)
    _randomise(ar)
    x0 = torch.randn(2, 5, D)
    step = _steps(n_layer)
    rec, inputs, _ = _recording(step)
    with torch.no_grad():
        got = ar(x0, rec)
        ref_inputs, _, ref_final = _reference_forward(ar, x0, step)
    for l in range(n_layer):
        torch.testing.assert_close(inputs[l], ref_inputs[l], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(got, ref_final, atol=1e-6, rtol=1e-6)


def test_zero_queries_give_uniform_weights_and_the_mean_of_the_sources():
    torch.manual_seed(0)
    n_layer, n_blocks = 6, 3
    ar = BlockAttnRes(D, n_layer=n_layer, n_blocks=n_blocks)
    x0 = torch.randn(2, 5, D)
    rec, inputs, outs = _recording(_steps(n_layer))
    with torch.no_grad():
        final = ar(x0, rec)
    for l in range(n_layer):
        srcs = _reference_sources(l, x0, outs, ar.layers_per_block)
        _, alpha = ar.mix(l, srcs)
        assert torch.equal(alpha, torch.full_like(alpha, 1.0 / len(srcs)))
        torch.testing.assert_close(inputs[l], torch.stack(srcs).mean(0), atol=1e-6, rtol=0)
    # Layer 0 sees only the embedding, so its input IS the embedding.
    assert torch.equal(inputs[0], x0)
    blocks = [x0] + [sum(outs[b * 2:(b + 1) * 2]) for b in range(n_blocks)]
    torch.testing.assert_close(final, torch.stack(blocks).mean(0), atol=1e-6, rtol=0)


def test_mixing_is_per_position_so_later_tokens_never_reach_earlier_ones():
    torch.manual_seed(0)
    ar = BlockAttnRes(D, n_layer=4, n_blocks=2)
    _randomise(ar)
    step = _steps(4)
    x0 = torch.randn(1, 8, D)
    x1 = x0.clone()
    x1[0, 5] += 1.0
    with torch.no_grad():
        a, b = ar(x0, step), ar(x1, step)
    assert torch.equal(a[0, :5], b[0, :5])
    assert not torch.allclose(a[0, 5], b[0, 5])


def test_gradients_reach_every_query_that_has_a_choice_and_the_embedding():
    torch.manual_seed(0)
    ar = BlockAttnRes(D, n_layer=4, n_blocks=2)
    x0 = torch.randn(2, 5, D, requires_grad=True)
    ar(x0, _steps(4)).pow(2).sum().backward()
    assert x0.grad is not None and x0.grad.abs().sum() > 0
    # Layer 0 has one source: softmax over one element is constant, no gradient.
    assert ar.queries[0].grad is None or torch.count_nonzero(ar.queries[0].grad) == 0
    for l in range(1, 5):
        assert ar.queries[l].grad is not None and ar.queries[l].grad.abs().sum() > 0, l


def test_bf16_autocast_keeps_the_mixing_in_the_sources_dtype():
    ar = BlockAttnRes(D, n_layer=2, n_blocks=2)
    x0 = torch.randn(2, 3, D)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = ar(x0, lambda l, h: torch.nn.functional.linear(h, torch.eye(D)))
    assert out.dtype == torch.float32 and torch.isfinite(out).all()
