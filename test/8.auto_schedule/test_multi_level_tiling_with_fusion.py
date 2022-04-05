import ir
import numpy as np

target = ir.CPU()
device = ir.Device(target)


def test_fusion():
    a = 128
    b = 256
    m = 4

    @ir.transform
    def test(w, x, y, z, u):
        ir.declare_var(w, (m, m, a, b), "int32", "input", "cpu")
        ir.declare_var(x, (m, m, b, a), "int32", "input", "cpu")
        ir.declare_var(y, (1, 1, a, a), "int32", "cache", "cpu")
        ir.declare_var(z, (m, m, a, a), "int32", "output", "cpu")
        "nid: L1"
        for i in range(m):
            "nid: L2"
            for j in range(m):
                "nid: L3"
                for k in range(b):
                    "nid: L4"
                    for p in range(a):
                        "nid: L5"
                        for q in range(a):
                            y[1, 1, p,
                              q] = y[1, 1, p, q] + w[i, j, p, k] * x[i, j, k, q]
                "nid: L6"
                for p in range(a):
                    "nid: L7"
                    for q in range(a):
                        z[i, j, p, q] = y[1, 1, p, q]

    s = ir.Schedule(test)
    print(s.ast())
    s = ir.AutoSchedule(s, target, device, 8)
    sch = s.test_multi_level_tiling_with_fusion(1)
    std_log = [
        'split(L4, factor=3, nparts=-1)', 'split(L4.0, factor=3, nparts=-1)',
        'split(L4.0.0, factor=3, nparts=-1)', 'split(L5, factor=3, nparts=-1)',
        'split(L5.0, factor=3, nparts=-1)',
        'split(L5.0.0, factor=3, nparts=-1)', 'split(L3, factor=16, nparts=-1)',
        'reorder(L4.0.0.0, L5.0.0.0, L4.0.0.1, L5.0.0.1, L3.0, L4.0.1, L5.0.1, L3.1, L4.1, L5.1)',
        'split(L6, factor=9, nparts=-1)', 'split(L6.0, factor=3, nparts=-1)',
        'split(L7, factor=9, nparts=-1)', 'split(L7.0, factor=3, nparts=-1)',
        'reorder(L6.0.0, L7.0.0, L6.0.1, L7.0.1, L6.1, L7.1)',
        'fuse(L4.0.0.0, L6.0.0)', 'fuse(L5.0.0.0, L7.0.0)',
        'fuse(L4.0.0.1, L6.0.1)', 'fuse(L5.0.0.1, L7.0.1)', 'cache(#23, y)'
    ]
    sch_log = sch.logs()
    assert std_log[:-1] == sch_log[:-1]
    assert sch_log[-1][:6] == 'cache(' and sch_log[-1][-4:] == ', y)'