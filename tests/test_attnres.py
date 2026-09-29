import pytest
import torch

from quipu.attnres import BlockAttnRes

D = 16


def _steps(n_steps: int, seed: int = 0):
    """A deterministic, per-token, nonlinear stand-in for sub-layer (step) l."""
    g = torch.Generator().manual_seed(seed)
    mats = [torch.randn(D, D, generator=g) / D ** 0.5 for _ in range(n_steps)]
    return lambda l, h: torch.tanh(h @ mats[l])


def _reference_sources(l: int, x0, outs, steps_per_block):
    """Kimi K3 source list for step l, written out by block index, independent of
    BlockAttnRes: b_0 = embedding; completed blocks are sums of their step outputs;
    after the first step of a block, the block's partial sum is appended."""
    n, i = divmod(l, steps_per_block)          # block n (0-based), position i in it
    srcs = [x0]
    for b in range(n):
        srcs.append(sum(outs[b * steps_per_block:(b + 1) * steps_per_block]))
    if i > 0:
        srcs.append(sum(outs[n * steps_per_block:l]))
    return srcs


def _reference_forward(ar: BlockAttnRes, x0, step):
    """AttnRes by the reference source lists; returns (inputs, outputs, final)."""
    lpb = ar.steps_per_block
    inputs, outs = [], []
    for l in range(ar.n_steps):
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
    alpha = (k @ ar.queries[ar.n_steps]).softmax(0)
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


def test_pseudo_queries_start_at_zero_one_per_step_plus_the_head():
    ar = BlockAttnRes(D, n_steps=6, n_blocks=3)
    assert len(ar.queries) == 7
    assert all(q.shape == (D,) and torch.count_nonzero(q) == 0 for q in ar.queries)


@pytest.mark.parametrize("n_steps, n_blocks", [(4, 3), (0, 1), (4, 0)])
def test_rejects_steps_that_do_not_split_into_blocks(n_steps, n_blocks):
    with pytest.raises(ValueError):
        BlockAttnRes(D, n_steps=n_steps, n_blocks=n_blocks)


@pytest.mark.parametrize("n_steps, n_blocks", [(6, 3), (6, 2), (4, 4), (4, 1)])
def test_matches_the_reference_source_lists_with_learned_queries(n_steps, n_blocks):
    torch.manual_seed(0)
    ar = BlockAttnRes(D, n_steps=n_steps, n_blocks=n_blocks)
    _randomise(ar)
    x0 = torch.randn(2, 5, D)
    step = _steps(n_steps)
    rec, inputs, _ = _recording(step)
    with torch.no_grad():
        got = ar(x0, rec)
        ref_inputs, _, ref_final = _reference_forward(ar, x0, step)
    # The reference stacks and normalises; the module scales dot products by a cached
    # inverse RMS and accumulates. Same math, different rounding: 1e-5.
    for l in range(n_steps):
        torch.testing.assert_close(inputs[l], ref_inputs[l], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(got, ref_final, atol=1e-5, rtol=1e-5)


def test_zero_queries_give_uniform_weights_and_the_mean_of_the_sources():
    torch.manual_seed(0)
    n_steps, n_blocks = 6, 3
    ar = BlockAttnRes(D, n_steps=n_steps, n_blocks=n_blocks)
    x0 = torch.randn(2, 5, D)
    rec, inputs, outs = _recording(_steps(n_steps))
    with torch.no_grad():
        final = ar(x0, rec)
    for l in range(n_steps):
        srcs = _reference_sources(l, x0, outs, ar.steps_per_block)
        _, alpha = ar.mix(l, srcs)
        assert torch.equal(alpha, torch.full_like(alpha, 1.0 / len(srcs)))
        torch.testing.assert_close(inputs[l], torch.stack(srcs).mean(0), atol=1e-6, rtol=0)
    # Layer 0 sees only the embedding, so its input IS the embedding.
    assert torch.equal(inputs[0], x0)
    blocks = [x0] + [sum(outs[b * 2:(b + 1) * 2]) for b in range(n_blocks)]
    torch.testing.assert_close(final, torch.stack(blocks).mean(0), atol=1e-6, rtol=0)


def test_mixing_is_per_position_so_later_tokens_never_reach_earlier_ones():
    torch.manual_seed(0)
    ar = BlockAttnRes(D, n_steps=4, n_blocks=2)
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
    ar = BlockAttnRes(D, n_steps=4, n_blocks=2)
    x0 = torch.randn(2, 5, D, requires_grad=True)
    ar(x0, _steps(4)).pow(2).sum().backward()
    assert x0.grad is not None and x0.grad.abs().sum() > 0
    # Layer 0 has one source: softmax over one element is constant, no gradient.
    assert ar.queries[0].grad is None or torch.count_nonzero(ar.queries[0].grad) == 0
    for l in range(1, 5):
        assert ar.queries[l].grad is not None and ar.queries[l].grad.abs().sum() > 0, l


def test_bf16_autocast_keeps_the_mixing_in_the_sources_dtype():
    ar = BlockAttnRes(D, n_steps=2, n_blocks=2)
    x0 = torch.randn(2, 3, D)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = ar(x0, lambda l, h: torch.nn.functional.linear(h, torch.eye(D)))
    assert out.dtype == torch.float32 and torch.isfinite(out).all()


def _naive_mix(query, sources, eps):
    """The textbook formula: stack, RMS-normalise the keys, softmax, weighted sum."""
    v = torch.stack(sources)
    k = torch.nn.functional.rms_norm(v, (v.shape[-1],), eps=eps)
    alpha = (k @ query).softmax(0)
    return (alpha[..., None] * v).sum(0), alpha


@pytest.mark.parametrize("m", [1, 2, 5])
def test_mix_matches_the_naive_stacked_formula_with_and_without_cached_rms(m):
    torch.manual_seed(0)
    ar = BlockAttnRes(D, n_steps=4, n_blocks=2)
    _randomise(ar)
    srcs = [torch.randn(3, 7, D) * (j + 1) for j in range(m)]
    want_h, want_a = _naive_mix(ar.queries[2], srcs, ar.eps)
    with torch.no_grad():
        h, alpha = ar.mix(2, srcs)
        h_cached, alpha_cached = ar.mix(2, srcs, [ar.inv_rms(s) for s in srcs[:-1]])
    for got_h, got_a in ((h, alpha), (h_cached, alpha_cached)):
        torch.testing.assert_close(got_h, want_h, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(got_a, want_a, atol=1e-6, rtol=1e-6)


def test_mix_keeps_no_stacked_or_normalised_copies_of_the_sources_for_backward():
    """Autograd may keep the sources themselves (they are alive anyway) and
    per-position scalars, but nothing of size m x (..., d)."""
    torch.manual_seed(0)
    ar = BlockAttnRes(D, n_steps=4, n_blocks=2)
    _randomise(ar)
    srcs = [torch.randn(4, 64, D, requires_grad=True) for _ in range(5)]
    src_ptrs = {s.untyped_storage().data_ptr() for s in srcs}
    big = []

    def pack(t):
        if t.numel() >= 4 * 64 * D and t.untyped_storage().data_ptr() not in src_ptrs:
            big.append(tuple(t.shape))
        return t
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        h, _ = ar.mix(3, srcs)
    assert big == []
    h.sum().backward()
    assert all(s.grad is not None for s in srcs)


def test_checkpointed_mix_gives_the_same_output_and_gradients():
    torch.manual_seed(0)
    x0 = torch.randn(2, 5, D)
    results = []
    for ckpt in (False, True):
        ar = BlockAttnRes(D, n_steps=6, n_blocks=3, checkpoint=ckpt).train()
        _randomise(ar)
        x = x0.clone().requires_grad_(True)
        out = ar(x, _steps(6))
        out.pow(2).sum().backward()
        results.append((out.detach(), x.grad, [q.grad for q in ar.queries]))
    (o0, g0, q0), (o1, g1, q1) = results
    assert torch.equal(o0, o1)
    torch.testing.assert_close(g1, g0, atol=1e-6, rtol=1e-6)
    for a, b in zip(q0[1:], q1[1:]):
        torch.testing.assert_close(b, a, atol=1e-6, rtol=1e-6)
