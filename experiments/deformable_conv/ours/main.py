import sys
import time
import math
import numpy as np
import ir
import ir.debug
from ir.libop import *

sys.path.append('../..')
from common.gpu import profile_start, profile_stop


def compile_all(n, c_in, c_out, h, w, k_h, k_w, device):
    mtype = device.main_mem_type()

    # yapf: disable

    @ir.transform
    def inference(X, W1, W2, Y):
        ir.declare_var(X, (n, c_in, h, w), "float32", "input", mtype)
        ir.declare_var(W1, (k_h, k_w, 2, c_in, k_h, k_w), "float32", "input", mtype)
        ir.declare_var(W2, (c_out, c_in, k_h, k_w), "float32", "input", mtype)
        ir.declare_var(Y, (n, c_out, h, w), "float32", "output", mtype)

        "nid: Li"
        for i in range(n):
            "nid: Lp"
            for p in range(h):
                "nid: Lq"
                for q in range(w):
                    pos = ir.create_var((k_h, k_w, 2), "float32", mtype)
                    pos_int = ir.create_var((k_h, k_w, 2), "int32", mtype)
                    "nid: Lro0"
                    for ro in range(k_h):
                        for so in range(k_w):
                            for t in range(2):
                                pos[ro, so, t] = 0
                            for ki in range(c_in):
                                for ri in range(k_h):
                                    for si in range(k_w):
                                        if p + ri >= 0 and p + ri < h and q + si >= 0 and q + si < w:
                                            for t in range(2):
                                                pos[ro, so, t] += X[i, ki, p + ri, q + si] * W1[ro, so, t, ki, ri, si]
                            for t in range(2):
                                pos[ro, so, t] /= c_in
                                pos_int[ro, so, t] = ir.cast(ir.floor(pos[ro, so, t]), "int32")

                    "nid: pixel"
                    pixel = ir.create_var((c_in, k_h, k_w), "float32", mtype)
                    for ki in range(c_in):
                        for ro in range(k_h):
                            for so in range(k_w):
                                x = ir.create_var((2, 2), "int32", mtype)
                                y = ir.create_var((2, 2), "int32", mtype)
                                x[0, 0] = p + ro + pos_int[ro, so, 0]
                                y[0, 0] = q + so + pos_int[ro, so, 1]
                                x[0, 1] = p + ro + pos_int[ro, so, 0]
                                y[0, 1] = q + so + pos_int[ro, so, 1] + 1
                                x[1, 0] = p + ro + pos_int[ro, so, 0] + 1
                                y[1, 0] = q + so + pos_int[ro, so, 1]
                                x[1, 1] = p + ro + pos_int[ro, so, 0] + 1
                                y[1, 1] = q + so + pos_int[ro, so, 1] + 1
                                dist = ir.create_var((2, 2), "float32", mtype)
                                dist[0, 0] = (pos[ro, so, 0] - pos_int[ro, so, 0]) * (pos[ro, so, 1] - pos_int[ro, so, 1])
                                dist[0, 1] = (pos[ro, so, 0] - pos_int[ro, so, 0]) * (pos_int[ro, so, 1] + 1 - pos[ro, so, 1])
                                dist[1, 0] = (pos_int[ro, so, 0] + 1 - pos[ro, so, 0]) * (pos[ro, so, 1] - pos_int[ro, so, 1])
                                dist[1, 1] = (pos_int[ro, so, 0] + 1 - pos[ro, so, 0]) * (pos_int[ro, so, 1] + 1 - pos[ro, so, 1])
                                pixel[ki, ro, so] = 0
                                for t in range(2):
                                    for u in range(2):
                                        if x[t, u] >= 0 and x[t, u] < h and y[t, u] >= 0 and y[t, u] < w:
                                            pixel[ki, ro, so] += X[i, ki, x[t, u], y[t, u]] * dist[t, u]

                    "nid: Lko"
                    einsum_("krs,lkrs->l")(pixel, W2, Y[i, :, p, q])

    # yapf: enable

    forward, backward, requires, privdes, _ = ir.grad(inference,
                                                      set(["X", "W1", "W2"]),
                                                      set(["Y"]))

    print("# Inference:")
    print(inference)
    s = ir.Schedule(inference)
    if device.target().type() == ir.TargetType.CPU:
        Lko = s.move_to("Lko-><staging:f_einsum>:5", ir.MoveToSide.After, "Li")
        _, _, _, Y_t_def = s.cache(Lko, "Y", "cpu")
        s.var_reorder(Y_t_def, [0, 2, 3, 1])
    else:
        Lko = s.move_to("Lko-><staging:f_einsum>:5", ir.MoveToSide.After, "Li")
        _, _, _, Y_t_def = s.cache(Lko, "Y", "gpu/global")
        s.var_reorder(Y_t_def, [0, 2, 3, 1])
        s.var_reorder("pixel", [3, 4, 5, 0, 1, 2])
    s.auto_schedule(device.target())
    f = ir.lower(s.func(), device.target())
    print(f)
    code = ir.codegen(f, device.target())
    print(ir.debug.with_line_no(code))
    inference_exe = ir.Driver(inference, code, device)

    print("# Forward:")
    print(forward)
    s = ir.Schedule(forward)
    if device.target().type() == ir.TargetType.CPU:
        Lko = s.move_to("Lko-><staging:f_einsum>:5", ir.MoveToSide.After, "Li")
        _, _, _, Y_t_def = s.cache(Lko, "Y", "cpu")
        s.var_reorder(Y_t_def, [0, 2, 3, 1])
    else:
        Lko = s.move_to("Lko-><staging:f_einsum>:5", ir.MoveToSide.After, "Li")
        _, _, _, Y_t_def = s.cache(Lko, "Y", "gpu/global")
        s.var_reorder(Y_t_def, [0, 2, 3, 1])
        s.var_reorder("pixel", [3, 4, 5, 0, 1, 2])
    s.auto_schedule(device.target())
    f = ir.lower(s.func(), device.target())
    print(f)
    code = ir.codegen(f, device.target())
    print(ir.debug.with_line_no(code))
    forward_exe = ir.Driver(forward, code, device)

    print("# Backward:")
    print(backward)
    s = ir.Schedule(backward)
    if device.target().type() == ir.TargetType.CPU:
        s.cache_reduction("Lro0", "X.grad", "cpu")
        s.fission("Li", ir.FissionSide.Before, "Lko-><staging:f_einsum>:5",
                  ".fwd", "")
        Lko = s.move_to("Lko-><staging:f_einsum>:5", ir.MoveToSide.Before, "Li")
        _, _, _, Y_t_def = s.cache(Lko, "Y.grad", "cpu")
        s.var_reorder(Y_t_def, [0, 2, 3, 1])
    else:
        #s.cache_reduction("Lro0", "X.grad", "gpu/global")
        s.fission("Li", ir.FissionSide.Before, "Lko-><staging:f_einsum>:5",
                  ".fwd", "")
        Lko = s.move_to("Lko-><staging:f_einsum>:5", ir.MoveToSide.Before, "Li")
        _, _, _, Y_t_def = s.cache(Lko, "Y.grad", "gpu/global")
        s.var_reorder(Y_t_def, [0, 2, 3, 1])
        s.var_reorder("pixel", [3, 4, 5, 0, 1, 2])
        s.var_reorder("pixel.grad", [3, 4, 5, 0, 1, 2])
    s.auto_schedule(device.target())
    f = ir.lower(s.func(), device.target())
    print(f)
    code = ir.codegen(f, device.target())
    print(ir.debug.with_line_no(code))
    backward_exe = ir.Driver(backward, code, device)

    def run_backward(x, w1, w2, y, d_y, d_x, d_w1, d_w2):
        kvs = {}
        kvs[privdes['Y']] = d_y
        kvs[requires['X']] = d_x
        kvs[requires['W1']] = d_w1
        kvs[requires['W2']] = d_w2
        backward_exe(x, w1, w2, y, **kvs)

    return inference_exe, forward_exe, run_backward


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <cpu/gpu>")
        exit(-1)
    device = sys.argv[1]

    n = 8
    c_in = 256
    c_out = 256
    h = 56
    w = 56
    k_h = 3
    k_w = 3
    x = np.random.uniform(size=(n, c_in, h, w)).astype("float32") * 2 - 1
    w1 = np.random.uniform(size=(k_h, k_w, 2, c_in, k_h,
                                 k_w)).astype("float32") * 2 - 1
    w2 = np.random.uniform(size=(c_out, c_in, k_h,
                                 k_w)).astype("float32") * 2 - 1
    y = np.zeros((n, c_out, h, w), dtype="float32")
    d_x = np.zeros(x.shape, dtype='float32')
    d_w1 = np.zeros(w1.shape, dtype='float32')
    d_w2 = np.zeros(w2.shape, dtype='float32')
    d_y = np.random.uniform(size=y.shape).astype('float32')

    if device == 'gpu':
        ir_dev = ir.Device(ir.GPU())
    else:
        assert device == 'cpu'
        ir_dev = ir.Device(ir.CPU())

    x = ir.Array(x, ir_dev)
    w1 = ir.Array(w1, ir_dev)
    w2 = ir.Array(w2, ir_dev)
    y = ir.Array(y, ir_dev)
    d_x = ir.Array(d_x, ir_dev)
    d_w1 = ir.Array(d_w1, ir_dev)
    d_w2 = ir.Array(d_w2, ir_dev)
    d_y = ir.Array(d_y, ir_dev)

    inference, forward, backward = compile_all(n, c_in, c_out, h, w, k_h, k_w,
                                               ir_dev)

    warmup_num = 10
    test_num = 100

    for i in range(warmup_num):
        inference(x, w1, w2, y)
    ir_dev.sync()
    t0 = time.time()
    for i in range(test_num):
        inference(x, w1, w2, y)
    ir_dev.sync()
    t1 = time.time()

    print(f"Inference Time = {(t1 - t0) / test_num * 1000} ms")

    for i in range(warmup_num):
        forward(x, w1, w2, y)
    ir_dev.sync()
    t0 = time.time()
    for i in range(test_num):
        forward(x, w1, w2, y)
    ir_dev.sync()
    t1 = time.time()

    print(f"Forward Time = {(t1 - t0) / test_num * 1000} ms")

    for i in range(warmup_num):
        backward(x, w1, w2, y, d_y, d_x, d_w1, d_w2)
    ir_dev.sync()
    #profile_start()
    t0 = time.time()
    for i in range(test_num):
        backward(x, w1, w2, y, d_y, d_x, d_w1, d_w2)
    ir_dev.sync()
    t1 = time.time()
    #profile_stop()

    print(f"Backward Time = {(t1 - t0) / test_num * 1000} ms")
