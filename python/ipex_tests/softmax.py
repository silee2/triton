
from ctypes import c_void_p, c_long
import torch
import random
from torch import empty_strided, as_strided, device
from torch._inductor.codecache import AsyncCompile

aten = torch.ops.aten
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
async_compile = AsyncCompile()

import triton
import triton.language as tl
from torch._inductor.triton_ops.autotune import grid
# from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


kernel0 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[16, 32],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 'xpu:0', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16
    rnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    x0 = xindex
    _tmp1 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + float("-inf")
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (32*x0)), xmask & rmask, eviction_policy='evict_last')
        _tmp1 = tl.where(xmask & rmask & (_tmp1 < tmp0), tmp0, _tmp1)
    tmp1 = tl.reshape(tl.max(_tmp1, 1), [XBLOCK, 1])
    _tmp5 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp2 = tl.load(in_ptr0 + (r1 + (32*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp3 = tmp2 * tmp1
        tmp4 = tl.exp(tmp3)
        _tmp5 = tl.where(xmask & rmask, _tmp5 + tmp4, _tmp5)
    tmp5 = tl.reshape(tl.sum(_tmp5, 1), [XBLOCK, 1])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(in_ptr0 + (r1 + (32*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp7 = tmp6 * tmp1
        tmp8 = tl.exp(tmp7)
        tmp9 = tmp8 / tmp5
        tl.store(out_ptr2 + (r1 + (32*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp9, xmask & rmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, = args
    args.clear()
    buf2 = empty_strided((16, 32), (32, 1), device='xpu', dtype=torch.float32)
    stream0 = torch.xpu.current_stream(0)
    kernel0.run(arg0_1, buf2, 16, 32, grid=grid(16), stream=stream0)
    del arg0_1
    return (buf2, )


if __name__ == "__main__":
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((16, 32), (32, 1), device='xpu', dtype=torch.float32)
    print_performance(lambda: call([arg0_1]))
