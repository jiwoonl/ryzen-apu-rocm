# SPDX-License-Identifier: MIT
"""Row-partitioned two-matmul dataflow; hidden activations stay in core SRAM.

Each of 16 cores owns m rows and all hidden columns. A hidden row block is
kept in a private SRAM Buffer in the second matmul's native blocked layout.
Only A, B, D and final C are DMA operands; no external hidden tensor exists.
"""
import numpy as np
from ml_dtypes import bfloat16
import aie.iron as iron
from aie.iron import Buffer, CompileTime, In, Out, ObjectFifo, Program, Runtime, Worker, kernels
from aie.iron.kernel import ExternalFunction
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorAccessPattern as TAP
from aie.utils import config

@iron.jit(aiecc_flags=['--alloc-scheme=basic-sequential'])
def sram_mlp(A:In, B:In, D:In, C:Out, *, batch:CompileTime[int]=1,
             m:CompileTime[int]=32, tile:CompileTime[int]=32,
             depth:CompileTime[int]=2):
    N=512; k=n=tile; M=batch*N
    assert m in (8,16,32) and tile in (16,32,64) and depth in (1,2,3)
    assert M%(16*m)==0
    ty=lambda count:np.ndarray[(count,),np.dtype[bfloat16]]
    mm=kernels.mm(dim_m=m,dim_k=k,dim_n=n,input_dtype=bfloat16,output_dtype=bfloat16)
    r,s,t=mm.mac_dims
    assert (r,s,t)==(4,8,4)
    source=r'''
#include <aie_api/aie.hpp>
extern "C" void original_mm(const bfloat16*, const bfloat16*, bfloat16*) asm("MM_SYMBOL");
extern "C" void save_hidden(bfloat16 *c, bfloat16 *hidden, int block) {
  // C is [m/4,n/4,4,4]; hidden tiles are [m/4,n/8,4,8].
  for (int rb=0; rb<TILE_M/4; ++rb)
    for (int cb=0; cb<TILE_N/8; ++cb) {
      auto a=aie::max(aie::load_v<16>(c+(rb*(TILE_N/4)+2*cb)*16),bfloat16(0));
      auto b=aie::max(aie::load_v<16>(c+(rb*(TILE_N/4)+2*cb+1)*16),bfloat16(0));
      auto [lo,hi]=aie::interleave_zip(a,b,4);
      bfloat16 *dst=hidden+block*TILE_M*TILE_N+(rb*(TILE_N/8)+cb)*32;
      aie::store_v(dst,lo); aie::store_v(dst+16,hi);
    }
}
extern "C" void hidden_mm(bfloat16 *hidden, bfloat16 *b, bfloat16 *c, int block) {
  original_mm(hidden+block*TILE_M*TILE_N,b,c);
}
'''.replace('MM_SYMBOL',mm._name)
    flags=[f'-DTILE_M={m}',f'-DTILE_N={n}']
    save=ExternalFunction('save_hidden',source_string=source,arg_types=[ty(m*n),ty(m*N),np.int32],include_dirs=[config.cxx_header_path()],compile_flags=flags)
    # Same source object exports the companion function without recompiling.
    from aie.iron.kernel import Kernel
    hidden_mm=Kernel('hidden_mm',save.object_file_name,[ty(m*N),ty(k*n),ty(m*n),np.int32])
    a3=[];a1=[];b3=[];b1=[];c3=[];c1=[]
    for col in range(4):
        aa=ObjectFifo(ty(4*m*k),name=f'a3_{col}',depth=depth)
        ais=aa.cons().split([i*m*k for i in range(4)],obj_types=[ty(m*k)]*4,
             names=[f'a1_{col}_{i}' for i in range(4)],
             dims_to_stream=[[(m//r,r*k),(k//s,s),(r,k),(s,1)]]*4)
        bb=ObjectFifo(ty(k*n),name=f'b3_{col}',depth=depth)
        bis=bb.cons().forward(obj_type=ty(k*n),name=f'b1_{col}',dims_to_stream=[(k//s,s*n),(n//t,t),(s,n),(t,1)])
        cc=ObjectFifo(ty(4*m*n),name=f'c3_{col}',depth=depth,
             dims_to_stream=[(m//r,r*n),(r,t),(n//t,r*t),(t,1)])
        cis=cc.prod().join([i*m*n for i in range(4)],obj_types=[ty(m*n)]*4,
             names=[f'c1_{col}_{i}' for i in range(4)],depths=[depth]*4)
        a3.append(aa);a1.append(ais);b3.append(bb);b1.append(bis);c3.append(cc);c1.append(cis)
    groups=M//(16*m)
    def core(a,b,c,hidden,temp,zero,matmul,store,hm):
        for _ in range_(groups):
            for j in range_(N//n):
                zero(temp)
                for _ in range_(N//k):
                    av=a.acquire(1);bv=b.acquire(1)
                    matmul(av,bv,temp)
                    a.release(1);b.release(1)
                store(temp,hidden,j)
            for _ in range_(N//n):
                cv=c.acquire(1);zero(cv)
                for kk in range_(N//k):
                    bv=b.acquire(1)
                    hm(hidden,bv,cv,kk)
                    b.release(1)
                c.release(1)
    workers=[]
    for col in range(4):
        for row in range(4):
            workers.append(Worker(core,[a1[col][row].cons(),b1[col].cons(),c1[col][row].prod(),
                 Buffer(ty(m*N),name=f'hidden_{col}_{row}'),Buffer(ty(m*n),name=f'temp_{col}_{row}'),
                 mm.zero,mm,save,hidden_mm],stack_size=0xD00))
    rt=Runtime()
    with rt.sequence(ty(M*N),ty(N*N),ty(N*N),ty(M*N)) as (aa,bb,dd,cc):
        rt.start(*workers)
        for group in range(groups):
            tg=rt.task_group()
            for col in range(4):
                row=(group*16+col*4)*m
                atap=TAP((M,N),row*N,[N//n,N//k,4*m,k],[0,k,N,1])
                btap=TAP((N,N),0,[N//n,N//k,k,n],[n,k*N,N,1])
                rt.fill(a3[col].prod(),aa,tap=atap,task_group=tg,wait=True)
                rt.fill(b3[col].prod(),bb,tap=btap,task_group=tg,wait=True)
            rt.finish_task_group(tg)
            tg=rt.task_group()
            for col in range(4):
                row=(group*16+col*4)*m
                ctap=TAP((M,N),row*N,[N//n,4,m,n],[n,m*N,N,1])
                rt.drain(c3[col].cons(),cc,tap=ctap,task_group=tg,wait=True)
                rt.fill(b3[col].prod(),dd,tap=btap,task_group=tg)
            rt.finish_task_group(tg)
    return Program(iron.get_current_device(),rt).resolve_program()
