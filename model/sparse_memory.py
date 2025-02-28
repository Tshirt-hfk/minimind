import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel_NSA(
    q,
    k,
    v,
    i,
    o,
    lse,
    L,
    H: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    S: tl.constexpr,
    STEP: tl.constexpr
):
    i_l, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    bos = (i_b * L + i_l) * H + i_h

    p_q = bos * K + tl.range(0, K)
    b_q = tl.load(q + p_q)

    o_acc = tl.zeros([V], dtype=tl.float32)
    lse_acc = tl.full([], 0.0, dtype=tl.float32)
    score_m = tl.full([], float('-inf'), dtype=tl.float32)
    for start_s in range(0, S):
        p_kv_bos = tl.load(i + bos * S + start_s).to(tl.int32) * N 
        for start_n in (0, N, STEP):
            p_kv_s = p_kv_bos + start_n + tl.range(0, STEP)[:, None]
            p_k = p_kv_s * K + tl.range(0, K)[None, :]
            p_v = p_kv_s * V + tl.range(0, V)[None, :]
            b_k = tl.load(k + p_k, mask=p_kv_s < N, other=0.0)
            b_v = tl.load(v + p_v, mask=p_kv_s < N, other=0.0)
            score = tl.where(p_kv_s[:, 0] < N, tl.dot(b_k, b_q), float('-inf'))
            score_mp, score_m = score_m, tl.maximum(tl.max(score, axis=0), score_m)
            score_b = tl.exp(score_mp - score_m)
            score_e = tl.exp(score - score_m)
            o_acc = o_acc * score_b + tl.sum(b_v * score_e[:, None], axis=0)
            lse_acc = lse_acc * score_b + tl.sum(score_e, axis=0)
    b_o = o_acc / lse_acc
    b_lse = tl.log(lse_acc) + score_m
    p_o = o + bos * V + tl.range(0, V)
    p_lse = lse + bos
    tl.store(p_o, b_o.to(o.dtype.element_ty))
    tl.store(p_lse, b_lse.to(lse.dtype.element_ty))


@triton.jit
def _bwd_kernel_preprocess_NSA(
    o,
    do,
    delta,
    BV: tl.constexpr,
    V: tl.constexpr
):
    i_n = tl.program_id(0)
    o_d = tl.arange(0, BV)
    m_d = o_d < V

    b_o = tl.load(o + i_n * V + o_d, mask=m_d, other=0)
    b_do = tl.load(do + i_n * V + o_d, mask=m_d, other=0).to(tl.float32)
    b_delta = tl.sum(b_o * b_do)

    tl.store(delta + i_n, b_delta.to(delta.dtype.element_ty))


@triton.jit
def _bwd_kernel_dq_NSA(
    q,
    k,
    v,
    i,
    lse,
    delta,
    do,
    dq,
    L,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    S: tl.constexpr,
    N: tl.constexpr,
    STEP: tl.constexpr
):
    i_l, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    bos = (i_b * L + i_l) * H + i_h

    p_q = bos * K + tl.range(0, K)
    b_q = tl.load(q + p_q)

    p_do = bos * V + tl.range(0, V)
    b_do = tl.load(do + p_do)

    b_lse = tl.load(lse + bos)

    b_delta = tl.load(delta + bos)

    b_dq_acc = tl.zeros([K], dtype=tl.float32)
    for start_s in range(0, S):
        p_kv_bos = tl.load(i + bos * S + start_s).to(tl.int32) * N
        for start_n in (0, N, STEP):
            p_kv_s = p_kv_bos + start_n + tl.range(0, STEP)[:, None]
            p_k = p_kv_s * K + tl.range(0, K)[None, :]
            p_v = p_kv_s * V + tl.range(0, V)[None, :]
            b_k = tl.load(k + p_k, mask=p_kv_s < N, other=0.0)
            b_v = tl.load(v + p_v, mask=p_kv_s < N, other=0.0)
            score = tl.where(p_kv_s[:, 0] < N, tl.dot(b_k, b_q), float('-inf'))
            score_e = tl.exp(score - b_lse)
            b_dp = tl.dot(b_v, b_do)
            b_ds = score_e * (b_dp.to(tl.float32) - b_delta)
            b_dq_acc += tl.dot(b_ds.to(b_k.dtype), b_k)
    q_dp = dq + bos * K + tl.range(0, K)
    tl.store(q_dp, b_dq_acc.to(q.dtype.element_ty))


@triton.jit
def _bwd_kernel_dkv_NSA(
    q,
    k,
    v,
    i,
    lse,
    delta,
    do,
    dk,
    dv,
    L,
    H,
    K,
    V,
    S,
    N,
    STEP
):
   ...



class sparse_memory_func(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, i):
        B, L, H, K, N, V, S = *q.shape, k.shape[1], v.shape[-1], i.shape[-1]
        STEP = 256 if torch.cuda.get_device_capability()[0] >= 9 else 128
        o = torch.empty(B, L, H, V, dtype=q.dtype, device=q.device)
        lse = torch.empty(B, L, H, dtype=q.dtype, device=q.device)
        _fwd_kernel_NSA[(L, B * H, 1)](
            q,
            k,
            v,
            i,
            o,
            lse,
            L,
            H,
            N,
            K,
            V,
            S,
            STEP
        )
        ctx.save_for_backward(q, k, v, i, o, lse)
        ctx.STEP = STEP
        return o.to(q.dtype)

    @staticmethod
    def backward(ctx, do):
        q, k, v, i, o, lse = ctx.saved_tensors
        B, L, H, K, V, S, N, = *q.shape, v.shape[-1], i.shape[-1], k.shape[1]
        STEP = ctx.STEP
        BV = triton.next_power_of_2(V)
        delta = torch.empty(B, L, H, dtype=torch.float32)
        _bwd_kernel_preprocess_NSA[(delta.numel(),)](
            o,
            do,
            delta,
            BV,
            V
        )
        dq = torch.empty(B, L, H, K, dtype=q.dtype, device=q.device)
        _bwd_kernel_dq_NSA[(L, B * H, 1)](
            q,
            k,
            v,
            i,
            lse,
            delta,
            do,
            dq,
            L,
            H,
            K,
            V,
            S,
            N,
            STEP
        )

        dk = torch.empty(k.shape, dtype=k.dtype, device=q.device)
        dv = torch.empty(v.shape, dtype=v.dtype, device=q.device)
        _bwd_kernel_dkv_NSA[()](
            q,
            k,
            v,
            i,
            lse,
            delta,
            do,
            dk,
            dv,
            L,
            H,
            K,
            V,
            S,
            N,
            STEP
        )
        return dq.to(q), dk.to(q), dv.to(q), None
        

def sparse_memory(q, k, v, indices):
    """
    q: [B, L, H, K]
    k: [M, N, K]
    v: [M, N, V]
    indices: [B, L, H, K]
    """
    return sparse_memory_func.apply(q, k, v, indices)