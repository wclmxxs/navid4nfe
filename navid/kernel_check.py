"""Run on each H200 before loading weights; no model download needed."""
from __future__ import annotations


def check_sol_kernel():
    import torch
    import triton
    from h3_runtime.sparse_attention import _dense, _import_kernel

    triton.set_allocator(lambda size, alignment, stream: torch.empty(size, device="cuda", dtype=torch.uint8))
    _import_kernel()
    from sol_attn.triton_ref import sol_attn

    generator = torch.Generator(device="cuda").manual_seed(1729)
    # Cover partial 64-token tails, a full bucket and the following bucket.
    # Interleaved QKV matches the Ulysses receive buffer used by production.
    for tokens in (3999, 4096, 4101):
        packed = torch.randn((1, tokens, 1, 384), dtype=torch.bfloat16,
                             device="cuda", generator=generator)
        q, k, v = packed.split(128, dim=-1)
        got = sol_attn(q, k, v, tau=-1000.0, compile_bucket_size=4096)
        expected = _dense(q[0], k[0], v[0]).unsqueeze(0)
        torch.testing.assert_close(got, expected, atol=0.015, rtol=0.03)
        # Sparse output must also be invariant to extra descriptor capacity.
        # This checks centroid mass/partial-block masks, which the dense limit
        # alone does not exercise. A sink forces conditioning blocks exact.
        sparse = sol_attn(q, k, v, tau=1.0, sink_tokens=257, sink_start=0, compile_bucket_size=4096)
        reference = sol_attn(q, k, v, tau=1.0, sink_tokens=257, sink_start=0)
        torch.testing.assert_close(sparse, reference, atol=0.015, rtol=0.03)
    torch.cuda.synchronize()
    print(f"SOL_KERNEL_OK: rank={torch.cuda.current_device()} triton_tma_sm90; bucket=4096; dense+sparse tails", flush=True)


if __name__ == "__main__":
    check_sol_kernel()
