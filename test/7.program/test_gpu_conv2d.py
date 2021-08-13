import functools

import ir
import ir.debug
import pytest
import numpy as np

target = ir.GPU()
device = ir.Device(target)

host = ir.Device(ir.CPU())


def test_manual_static():
    # Matching http://tvm.apache.org/docs/tutorials/optimize/opt_conv_cuda.html

    batch = 256
    in_channel = 256
    out_channel = 512
    in_size = 14
    kernel = 3
    pad = 1
    stride = 1
    out_size = (in_size - kernel + 2 * pad) // stride + 1

    A_np = np.random.uniform(size=(in_size, in_size, in_channel,
                                   batch)).astype("float32")
    W_np = np.random.uniform(size=(kernel, kernel, in_channel,
                                   out_channel)).astype("float32")

    def eval(func, print_code=False, time=False):
        func = ir.lower(func, target)
        if print_code:
            print(func, flush=True)
        code = ir.codegen(func, target)
        if print_code:
            print(ir.debug.with_line_no(code), flush=True)
        driver = ir.Driver(func, code, device)
        B_np = np.zeros((out_size, out_size, out_channel, batch),
                        dtype="float32")
        A_arr = ir.Array(A_np, device)
        W_arr = ir.Array(W_np, device)
        B_arr = ir.Array(B_np, device)
        driver.set_params(A=A_arr, W=W_arr, B=B_arr)
        if time:
            t = driver.time()
            print("time: %s ms" % t)
        else:
            driver.run()
        B_np = B_arr.numpy().reshape(out_size, out_size, out_channel, batch)
        return B_np

    @ir.transform
    def algo(A, W, B):
        ir.declare_var(A, (in_size, in_size, in_channel, batch), "float32",
                       "input", "gpu/global")
        ir.declare_var(W, (kernel, kernel, in_channel, out_channel), "float32",
                       "input", "gpu/global")
        ir.declare_var(B, (out_size, out_size, out_channel, batch), "float32",
                       "output", "gpu/global")
        '''nid: Ly'''
        for yy in range(out_size):
            '''nid: Lx'''
            for xx in range(out_size):
                '''nid: Lf'''
                for ff in range(out_channel):
                    '''nid: Ln'''
                    for nn in range(batch):
                        '''nid: init'''
                        B[yy, xx, ff, nn] = 0.
                        '''nid: Lry'''
                        for ry in range(kernel):
                            '''nid: Lrx'''
                            for rx in range(0, kernel):
                                '''nid: Lrc'''
                                for rc in range(0, in_channel):
                                    # TODO: Let y = yy * stride + ry - pad
                                    # TODO：Let x = xx * stride + rx - pad
                                    if yy * stride + ry - pad >= 0 and yy * stride + ry - pad < in_size and xx * stride + rx - pad >= 0 and xx * stride + rx - pad < in_size:
                                        B[yy, xx, ff,
                                          nn] += A[yy * stride + ry - pad,
                                                   xx * stride + rx - pad, rc,
                                                   nn] * W[ry, rx, rc, ff]

    tile = 8
    num_thread = 8
    block_factor = tile * num_thread
    step = 8
    vthread = 2
    hi, wi, fi, ni, ryi, rxi, rci = "Ly", "Lx", "Lf", "Ln", "Lry", "Lrx", "Lrc"
    s = ir.Schedule(algo)

    bz = s.merge(hi, wi)
    by, fi = s.split(fi, factor=block_factor)
    bx, ni = s.split(ni, factor=block_factor)
    tyz, fi = s.split(fi, nparts=vthread)
    txz, ni = s.split(ni, nparts=vthread)
    ty, fi = s.split(fi, nparts=num_thread)
    tx, ni = s.split(ni, nparts=num_thread)
    s.reorder([bz, by, bx, ty, tx, tyz, txz, fi, ni])

    s.move_to("init", ir.MoveToSide.Before, fi)
    rco, rci = s.split(rci, factor=step)
    s.reorder([rco, ryi, rxi, rci, fi, ni])

    s.parallelize(bz, "blockIdx.z")
    s.parallelize(by, "blockIdx.y")
    s.parallelize(bx, "blockIdx.x")
    s.parallelize(ty, "threadIdx.y")
    s.parallelize(tx, "threadIdx.x")

    s.cache(s.find(lambda x: x.nid() == txz).node().body, "B", "gpu/local")

    fill_AA, _, AA = s.cache(rci, "A", "gpu/shared")
    fill_AA = s.find(lambda x: x.nid() == fill_AA)
    fill_AA_ty, fill_AA_tx = fill_AA.outer().outer(), fill_AA.outer()
    fill_AA_tx, fill_AA_vec = s.split(fill_AA_tx, factor=4)
    s.parallelize(fill_AA_ty, "threadIdx.y")
    s.parallelize(fill_AA_tx, "threadIdx.x")
    s.vectorize(fill_AA_vec)

    fill_WW, _, WW = s.cache(rci, "W", "gpu/shared")
    fill_WW = s.find(lambda x: x.nid() == fill_WW)
    fill_WW_ty, fill_WW_tx = fill_WW.outer().outer(), fill_WW.outer()
    fill_WW_tx, fill_WW_vec = s.split(fill_WW_tx, factor=4)
    s.parallelize(fill_WW_ty, "threadIdx.y")
    s.parallelize(fill_WW_tx, "threadIdx.x")
    s.vectorize(fill_WW_vec)

    fill_AL, _, AL = s.cache(fi, AA, "gpu/local")
    fill_WL, _, WL = s.cache(fi, WW, "gpu/local")

    s.blend(txz)
    s.blend(tyz)

    func = s.func()
    result = eval(func, True, True)  # Should be about 10ms on V100

    s = ir.Schedule(algo)
    s.parallelize("Ly", "blockIdx.y")
    s.parallelize("Lx", "blockIdx.x")
    s.parallelize("Lf", "threadIdx.x")
    baseline = s.func()
    result_std = eval(baseline)
    assert np.all(np.isclose(result, result_std))