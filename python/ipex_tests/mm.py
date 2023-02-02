import torch
import triton
import triton.language as tl

@triton.jit
def kernel0(
        A,
        B,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        # fusable kernels args
        out_ptr3,
        allow_tf32: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
        SPLIT_K: tl.constexpr,
        EVEN_K: tl.constexpr,
        ACC_TYPE: tl.constexpr,
):
    # matrix multiplication
    pid = tl.program_id(0)
    pid_z = tl.program_id(1)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    # do matrix multiplication
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = pid_z * BLOCK_K + tl.arange(0, BLOCK_K)
    # pointers
    A_ptrs = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B_ptrs = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(K, 0, -BLOCK_K * SPLIT_K):
        if EVEN_K:
            a = tl.load(A_ptrs)
            b = tl.load(B_ptrs)
        else:
            a = tl.load(A_ptrs, mask=rk[None, :] < k, other=0.0)
            b = tl.load(B_ptrs, mask=rk[:, None] < k, other=0.0)
        acc += tl.dot(a, b, allow_tf32=allow_tf32)
        A_ptrs += BLOCK_K * SPLIT_K * stride_ak
        B_ptrs += BLOCK_K * SPLIT_K * stride_bk
    acc = acc.to(out_ptr3.dtype.element_ty)


    XBLOCK: tl.constexpr = BLOCK_M
    YBLOCK: tl.constexpr = BLOCK_N
    xnumel = M
    ynumel = N
    xoffset = pid_m * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    yoffset = pid_n * YBLOCK
    yindex = yoffset + tl.reshape(tl.arange(0, YBLOCK), [1, YBLOCK])
    ymask = yindex < ynumel
    x0 = xindex
    y1 = yindex
    tmp0 = tl.maximum(0, acc)
    tl.store(out_ptr3 + (y1 + (1024*x0) + tl.zeros([XBLOCK, YBLOCK], tl.int32)), tmp0, xmask & ymask)


ret = triton.compile(kernel0, signature="*fp32,*fp32,*fp32", constants={"M": 1024, "N": 1024,  "K": 1024, "stride_am": 1024, "stride_ak": 1, "stride_bk": 1024, "stride_bn": 1, "stride_cm": 1024, "stride_cn": 1, "BLOCK_M": 8, "BLOCK_N": 8, "BLOCK_K": 8, "SPLIT_K": 8, "GROUP_M": 8, "ACC_TYPE": tl.float32, "allow_tf32": False}, output="ttgir")
print(ret)

