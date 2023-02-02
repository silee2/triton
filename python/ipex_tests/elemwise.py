import torch
from torch.testing import assert_close

import triton
import triton.language as tl

num_warps, block_size, iter_size = [4, 1024, 256]

@triton.jit
def kernel(X, Y, Z, N):
    pid = tl.program_id(0)
    idx= pid * 512 + tl.arange(0, 512)
    mask = idx < N
    x = tl.load(X + idx, mask=mask)
    y = tl.load(Y + idx, mask=mask)
    tl.store(Z + idx, x + y, mask=mask)

grid = lambda EA: (4096 // (block_size),)
k = kernel[grid]
ret = triton.compile(kernel, signature="*fp32,*fp32,*fp32", constants={"N": 4096}, output="ttgir")
print(ret)

