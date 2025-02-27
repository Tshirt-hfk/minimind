import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel_QM(
    Q,
    M,
    M_IDX,
    Out,
    stride_qb,
    stride_qh,
    stride_qc,
    stride_mm,
    stride_mn,
    stride_mc,
    stride_ib,
    stride_ih,
    stride_ik,
    stride_ob,
    stride_oh,
    stride_oc,
    K: tl.constexpr,
    N: tl.constexpr,
    C: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    start_m = tl.program_id(0)
    offs_c = tl.arange(0, C)
    offs_k = tl.arange(0, K)
    offs_n = tl.arange(0, BLOCK_N)
    off_q = (start_m * stride_qh + offs_c * stride_qc)
    off_i = (start_m * stride_ih + offs_k * stride_ik)
    q = tl.load(Q + off_q)
    off_m_idx = tl.load(M_IDX + off_i)
    out = tl.zeros([C], dtype=tl.float32)
    s_sum = 0
    s_sum_weight = 1
    for start_k in range(0, K):
        for start_n in (0, N, BLOCK_N):
            off_m = off_m_idx[start_k] * stride_mm + \
                    (start_n + offs_n[:, None]) * stride_mn + \
                    offs_c[None, :] * stride_mc
            m = tl.load(M + off_m, mask=start_n + offs_n[:, None] < N, other=0.0)
            s = tl.dot(q, tf.trans(m))
            s_max = tl.max(s, axis=0)
            s = s - s_max
            s_exp = tl.exp(s)
            

@triton.jit
def _bwd_kernel_QM(
    Q,
    M,
    Out
):
    ...


class sparse_memory_func(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, m, m_idx):
        """
        q: [B, H, C]
        m: [M, N, C]
        m_idx: [B, H, K]
        """
        capability = torch.cuda.get_device_capability()
        if capability[0] < 8:
            raise RuntimeError(
                "Sparse Memory currently only supported for compute capability >= 80"
            )
        B, H, C = q.shape
        M, N, C = m.shape
        B, H, K = m_idx.shape
        o = torch.empty(B, H, C, dtype=q.dtype, device=q.device)
        BLOCK_M = 128
        BLOCK_N = 64
        grid = (B * H, triton.cdiv(N*K, BLOCK_N), 1)
        ...

    @staticmethod
    def backward(ctx, dout):
        ...


def sparse_memory(q, m, m_idx):
    """
    q: [B, H, C]
    m: [M, N, C]
    m_idx: [B, H, K]
    """
    q = q.contiguous()
    m = m.contiguous()
    m_idx = m_idx.contiguous()
    return sparse_memory_func.apply(q, m, m_idx)