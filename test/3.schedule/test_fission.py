import ir
import pytest


def test_fission_after():
    with ir.VarDef([
        ("y", (4, 8), "int32", "output", "cpu"),
        ("z", (4, 8), "int32", "output", "cpu"),
    ]) as (y, z):
        with ir.For("i", 0, 4, nid="L1") as i:
            with ir.For("j", 0, 8, nid="L2") as j:
                ir.MarkNid("S0")
                y[i, j] = i + j
                z[i, j] = i * j
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    s.fission("L2", ir.FissionSide.After, "S0")
    ast = s.ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([
        ("y", (4, 8), "int32", "output", "cpu"),
        ("z", (4, 8), "int32", "output", "cpu"),
    ]) as (y, z):
        with ir.For("i", 0, 4) as i:
            with ir.For("j", 0, 8) as j:
                y[i, j] = i + j
            with ir.For("j", 0, 8) as j:
                z[i, j] = i * j
    std = ir.pop_ast()

    assert std.match(ast)


def test_fission_before():
    with ir.VarDef([
        ("y", (4, 8), "int32", "output", "cpu"),
        ("z", (4, 8), "int32", "output", "cpu"),
    ]) as (y, z):
        with ir.For("i", 0, 4, nid="L1") as i:
            with ir.For("j", 0, 8, nid="L2") as j:
                y[i, j] = i + j
                ir.MarkNid("S0")
                z[i, j] = i * j
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    s.fission("L2", ir.FissionSide.Before, "S0")
    ast = s.ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([
        ("y", (4, 8), "int32", "output", "cpu"),
        ("z", (4, 8), "int32", "output", "cpu"),
    ]) as (y, z):
        with ir.For("i", 0, 4) as i:
            with ir.For("j", 0, 8) as j:
                y[i, j] = i + j
            with ir.For("j", 0, 8) as j:
                z[i, j] = i * j
    std = ir.pop_ast()

    assert std.match(ast)


def test_fission_after_empty():
    with ir.VarDef("z", (4, 8), "int32", "output", "cpu") as z:
        with ir.For("i", 0, 4, nid="L1") as i:
            with ir.For("j", 0, 8, nid="L2") as j:
                ir.MarkNid("S0")
                z[i, j] = i * j
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    s.fission("L2", ir.FissionSide.After, "S0")
    ast = s.ast()
    print(ast)
    ast = ir.simplify_pass(ast)
    print(ast)

    with ir.VarDef("z", (4, 8), "int32", "output", "cpu") as z:
        with ir.For("i", 0, 4) as i:
            with ir.For("j", 0, 8) as j:
                z[i, j] = i * j
    std = ir.pop_ast()

    assert std.match(ast)


def test_fission_before_empty():
    with ir.VarDef("y", (4, 8), "int32", "output", "cpu") as y:
        with ir.For("i", 0, 4, nid="L1") as i:
            with ir.For("j", 0, 8, nid="L2") as j:
                ir.MarkNid("S0")
                y[i, j] = i + j
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    s.fission("L2", ir.FissionSide.Before, "S0")
    ast = s.ast()
    print(ast)
    ast = ir.simplify_pass(ast)
    print(ast)

    with ir.VarDef("y", (4, 8), "int32", "output", "cpu") as y:
        with ir.For("i", 0, 4) as i:
            with ir.For("j", 0, 8) as j:
                y[i, j] = i + j
    std = ir.pop_ast()

    assert std.match(ast)


def test_stmt_in_if():
    with ir.VarDef([
        ("y", (4, 8), "int32", "output", "cpu"),
        ("z", (4, 8), "int32", "output", "cpu"),
    ]) as (y, z):
        with ir.For("i", 0, 4, nid="L1") as i:
            with ir.For("j", 0, 8, nid="L2") as j:
                with ir.If(i > 1):
                    ir.MarkNid("S0")
                    y[i, j] = i + j
                z[i, j] = i * j
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    s.fission("L2", ir.FissionSide.After, "S0")
    ast = s.ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([
        ("y", (4, 8), "int32", "output", "cpu"),
        ("z", (4, 8), "int32", "output", "cpu"),
    ]) as (y, z):
        with ir.For("i", 0, 4) as i:
            with ir.If(i > 1):
                with ir.For("j", 0, 8) as j:
                    y[i, j] = i + j
            with ir.For("j", 0, 8) as j:
                z[i, j] = i * j
    std = ir.pop_ast()

    assert std.match(ast)


def test_buffer_hoist():
    with ir.VarDef([
        ("x0", (4, 8), "int32", "input", "cpu"),
        ("x1", (4, 8), "int32", "input", "cpu"),
        ("y", (4, 8), "int32", "output", "cpu"),
    ]) as (x0, x1, y):
        with ir.For("i", 0, 4, nid="L1") as i:
            with ir.For("j", 0, 8, nid="L2") as j:
                with ir.VarDef("buf", (8,), "int32", "cache", "cpu") as b:
                    ir.MarkNid("S0")
                    b[j] = x0[i, j] + x1[i, j]
                    y[i, j] = b[j] * b[j]
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    s.fission("L2", ir.FissionSide.After, "S0")
    ast = s.ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([
        ("x0", (4, 8), "int32", "input", "cpu"),
        ("x1", (4, 8), "int32", "input", "cpu"),
        ("y", (4, 8), "int32", "output", "cpu"),
    ]) as (x0, x1, y):
        with ir.For("i", 0, 4) as i:
            with ir.VarDef("buf", (8,), "int32", "cache", "cpu") as b:
                with ir.For("j", 0, 8) as j:
                    b[j] = x0[i, j] + x1[i, j]
                with ir.For("j", 0, 8) as j:
                    y[i, j] = b[j] * b[j]
    std = ir.pop_ast()

    assert std.match(ast)


def test_buffer_no_hoist():
    with ir.VarDef([
        ("x0", (4, 8), "int32", "input", "cpu"),
        ("x1", (4, 8), "int32", "input", "cpu"),
        ("y", (4, 8), "int32", "output", "cpu"),
        ("z", (4, 8), "int32", "output", "cpu"),
    ]) as (x0, x1, y, z):
        with ir.For("i", 0, 4, nid="L1") as i:
            with ir.For("j", 0, 8, nid="L2") as j:
                with ir.VarDef("buf", (4, 8), "int32", "cache", "cpu") as b:
                    b[i, j] = x0[i, j] + x1[i, j]
                    ir.MarkNid("S0")
                    y[i, j] = b[i, j] * b[i, j]
                    z[i, j] = x0[i, j] * 2
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    s.fission("L2", ir.FissionSide.After, "S0")
    ast = s.ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([
        ("x0", (4, 8), "int32", "input", "cpu"),
        ("x1", (4, 8), "int32", "input", "cpu"),
        ("y", (4, 8), "int32", "output", "cpu"),
        ("z", (4, 8), "int32", "output", "cpu"),
    ]) as (x0, x1, y, z):
        with ir.For("i", 0, 4) as i:
            # buf is not here
            with ir.For("j", 0, 8) as j:
                ir.Any()  # May be shrinked
            with ir.For("j", 0, 8) as j:
                z[i, j] = x0[i, j] * 2
    std = ir.pop_ast()

    assert std.match(ast)


def test_correct_dependency_after():
    with ir.VarDef([
        ("x0", (4, 8), "int32", "input", "cpu"),
        ("x1", (4, 8), "int32", "input", "cpu"),
        ("y", (4, 8), "int32", "output", "cpu"),
    ]) as (x0, x1, y):
        with ir.For("i", 0, 4, nid="L1") as i:
            with ir.For("j", 0, 8, nid="L2") as j:
                with ir.VarDef("buf", (1,), "int32", "cache", "cpu") as b:
                    ir.MarkNid("S0")
                    b[0] = x0[i, j] + x1[i, j]
                    y[i, j] = b[0] * b[0]
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    s.fission("L2", ir.FissionSide.After, "S0")
    ast = s.ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([
        ("x0", (4, 8), "int32", "input", "cpu"),
        ("x1", (4, 8), "int32", "input", "cpu"),
        ("y", (4, 8), "int32", "output", "cpu"),
    ]) as (x0, x1, y):
        with ir.For("i", 0, 4) as i:
            with ir.VarDef("buf", (8, 1), "int32", "cache", "cpu") as b:
                with ir.For("j", 0, 8) as j:
                    b[j, 0] = x0[i, j] + x1[i, j]
                with ir.For("j", 0, 8) as j:
                    y[i, j] = b[j, 0] * b[j, 0]
    std = ir.pop_ast()

    assert std.match(ast)


def test_correct_dependency_before():
    with ir.VarDef([
        ("x0", (4, 8), "int32", "input", "cpu"),
        ("x1", (4, 8), "int32", "input", "cpu"),
        ("y", (4, 8), "int32", "output", "cpu"),
    ]) as (x0, x1, y):
        with ir.For("i", 0, 4, nid="L1") as i:
            with ir.For("j", 0, 8, nid="L2") as j:
                with ir.VarDef("buf", (1,), "int32", "cache", "cpu") as b:
                    b[0] = x0[i, j] + x1[i, j]
                    ir.MarkNid("S0")
                    y[i, j] = b[0] * b[0]
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    s.fission("L2", ir.FissionSide.Before, "S0")
    ast = s.ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([
        ("x0", (4, 8), "int32", "input", "cpu"),
        ("x1", (4, 8), "int32", "input", "cpu"),
        ("y", (4, 8), "int32", "output", "cpu"),
    ]) as (x0, x1, y):
        with ir.For("i", 0, 4) as i:
            with ir.VarDef("buf", (8, 1), "int32", "cache", "cpu") as b:
                with ir.For("j", 0, 8) as j:
                    b[j, 0] = x0[i, j] + x1[i, j]
                with ir.For("j", 0, 8) as j:
                    y[i, j] = b[j, 0] * b[j, 0]
    std = ir.pop_ast()

    assert std.match(ast)


def test_correct_dependency_loop_step():
    with ir.VarDef([
        ("x0", (4, 8), "int32", "input", "cpu"),
        ("x1", (4, 8), "int32", "input", "cpu"),
        ("y", (4, 8), "int32", "output", "cpu"),
    ]) as (x0, x1, y):
        with ir.For("i", 0, 4, nid="L1") as i:
            with ir.For("j", 0, 8, 2, nid="L2") as j:
                with ir.VarDef("buf", (1,), "int32", "cache", "cpu") as b:
                    ir.MarkNid("S0")
                    b[0] = x0[i, j] + x1[i, j]
                    y[i, j] = b[0] * b[0]
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    s.fission("L2", ir.FissionSide.After, "S0")
    ast = s.ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([
        ("x0", (4, 8), "int32", "input", "cpu"),
        ("x1", (4, 8), "int32", "input", "cpu"),
        ("y", (4, 8), "int32", "output", "cpu"),
    ]) as (x0, x1, y):
        with ir.For("i", 0, 4) as i:
            with ir.VarDef("buf", (4, 1), "int32", "cache", "cpu") as b:
                with ir.For("j", 0, 8, 2) as j:
                    b[j // 2, 0] = x0[i, j] + x1[i, j]
                with ir.For("j", 0, 8, 2) as j:
                    y[i, j] = b[j // 2, 0] * b[j // 2, 0]
    std = ir.use_builtin_div(ir.pop_ast())

    assert std.match(ast)


def test_correct_dependency_multi_loop_1():
    with ir.VarDef([
        ("x0", (4, 8), "int32", "input", "cpu"),
        ("x1", (4, 8), "int32", "input", "cpu"),
        ("y", (4, 8), "int32", "output", "cpu"),
    ]) as (x0, x1, y):
        with ir.For("i", 0, 4, nid="L1") as i:
            with ir.For("j", 0, 8, nid="L2") as j:
                with ir.VarDef("buf", (1,), "int32", "cache", "cpu") as b:
                    ir.MarkNid("S0")
                    b[0] = x0[i, j] + x1[i, j]
                    y[i, j] = b[0] * b[0]
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    s.fission("L1", ir.FissionSide.After, "S0")
    ast = s.ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([
        ("x0", (4, 8), "int32", "input", "cpu"),
        ("x1", (4, 8), "int32", "input", "cpu"),
        ("y", (4, 8), "int32", "output", "cpu"),
    ]) as (x0, x1, y):
        with ir.VarDef("buf", (4, 8, 1), "int32", "cache", "cpu") as b:
            with ir.For("i", 0, 4) as i:
                with ir.For("j", 0, 8) as j:
                    b[i, j, 0] = x0[i, j] + x1[i, j]
            with ir.For("i", 0, 4) as i:
                with ir.For("j", 0, 8) as j:
                    y[i, j] = b[i, j, 0] * b[i, j, 0]
    std = ir.pop_ast()

    assert std.match(ast)


def test_correct_dependency_multi_loop_2():
    with ir.VarDef([("a", (4, 4), "float32", "input", "cpu"),
                    ("b", (4,), "float32", "input", "cpu"),
                    ("d_y", (4,), "float32", "inout", "cpu"),
                    ("d_a", (4, 4), "float32", "inout", "cpu"),
                    ("d_b", (4,), "float32", "inout", "cpu")]) as (a, b, d_y,
                                                                   d_a, d_b):
        with ir.For("i", 0, 4, nid="L1") as i:
            with ir.For("j", 0, 4) as j:
                with ir.VarDef("d_y_old", (), "float32", "cache",
                               "cpu") as d_y_old:
                    d_y_old[()] = d_y[i]
                    d_y[i] = 2 * d_y_old[()]
                    ir.MarkNid("S0")
                    d_a[i, j] += d_y_old[()] * b[j]
                    d_b[j] += d_y_old[()] * a[i, j]
            d_y[i] = 0
    ast = ir.make_reduction(ir.pop_ast())
    print(ast)
    s = ir.Schedule(ast)
    s.fission("L1", ir.FissionSide.Before, "S0")
    ast = s.ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("a", (4, 4), "float32", "input", "cpu"),
                    ("b", (4,), "float32", "input", "cpu"),
                    ("d_y", (4,), "float32", "inout", "cpu"),
                    ("d_a", (4, 4), "float32", "inout", "cpu"),
                    ("d_b", (4,), "float32", "inout", "cpu"),
                    ("d_y_old", (4, 4), "float32", "cache", "cpu")
                   ]) as (a, b, d_y, d_a, d_b, d_y_old):
        with ir.For("i", 0, 4) as i:
            with ir.For("j", 0, 4) as j:
                d_y_old[i, j] = d_y[i]
                d_y[i] = 2 * d_y_old[i, j]
        with ir.For("i", 0, 4) as i:
            with ir.For("j", 0, 4) as j:
                d_a[i, j] += d_y_old[i, j] * b[j]
                d_b[j] += d_y_old[i, j] * a[i, j]
            d_y[i] = 0
    std = ir.make_reduction(ir.pop_ast())

    assert std.match(ast)


def test_correct_dependency_real_dep():
    with ir.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4, 8), "int32", "output", "cpu")]) as (x, y):
        with ir.For("i", 0, 4, nid="L1") as i:
            with ir.VarDef("buf", (1,), "int32", "cache", "cpu") as b:
                ir.MarkNid("S0")
                b[0] = x[i] * 2
                with ir.For("j", 0, 8, nid="L2") as j:
                    y[i, j] = b[0] * b[0]
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    s.fission("L1", ir.FissionSide.After, "S0")
    ast = s.ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4, 8), "int32", "output", "cpu")]) as (x, y):
        with ir.VarDef("buf", (4, 1), "int32", "cache", "cpu") as b:
            with ir.For("i", 0, 4) as i:
                b[i, 0] = x[i] * 2
            with ir.For("i", 0, 4) as i:
                with ir.For("j", 0, 8) as j:
                    y[i, j] = b[i, 0] * b[i, 0]
    std = ir.pop_ast()

    assert std.match(ast)


def test_correct_dependency_unable_resolve():
    with ir.VarDef([
        ("x0", (4, 8), "int32", "input", "cpu"),
        ("x1", (4, 8), "int32", "input", "cpu"),
        ("y", (4, 8), "int32", "output", "cpu"),
        ("buf", (1,), "int32", "inout", "cpu"),
    ]) as (x0, x1, y, b):
        with ir.For("i", 0, 4, nid="L1") as i:
            with ir.For("j", 0, 8, nid="L2") as j:
                ir.MarkNid("S0")
                b[0] = x0[i, j] + x1[i, j]
                y[i, j] = b[0] * b[0]
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    with pytest.raises(ir.InvalidSchedule):
        s.fission("L2", ir.FissionSide.After, "S0")
    ast_ = s.ast()  # Should not changed
    assert ast_.match(ast)


def test_correct_dependency_no_need_to_modify_no_dep():
    with ir.VarDef([
        ("x0", (4, 4), "int32", "input", "cpu"),
        ("x1", (4, 4), "int32", "input", "cpu"),
        ("y", (4, 4, 4), "int32", "output", "cpu"),
    ]) as (x0, x1, y):
        with ir.For("i", 0, 4, nid="L1") as i:
            with ir.For("j", 0, 4, nid="L2") as j:
                with ir.VarDef("buf", (4,), "int32", "cache", "cpu") as b:
                    with ir.For("k", 0, 4, nid="L3") as k:
                        ir.MarkNid("S0")
                        b[k] = x0[i, k]
                        y[i, j, k] = b[k] * x1[i, j]
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    s.fission("L2", ir.FissionSide.After, "S0")
    ast = s.ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([
        ("x0", (4, 4), "int32", "input", "cpu"),
        ("x1", (4, 4), "int32", "input", "cpu"),
        ("y", (4, 4, 4), "int32", "output", "cpu"),
    ]) as (x0, x1, y):
        with ir.For("i", 0, 4) as i:
            with ir.VarDef("buf", (4,), "int32", "cache", "cpu") as b:
                with ir.For("k", 0, 4) as k:
                    b[k] = x0[i, k]
                with ir.For("j", 0, 4) as j:
                    with ir.For("k", 0, 4) as k:
                        y[i, j, k] = b[k] * x1[i, j]
    std = ir.pop_ast()

    assert std.match(ast)


def test_correct_dependency_no_need_to_modify_broadcast():
    with ir.VarDef([
        ("x0", (4,), "int32", "input", "cpu"),
        ("x1", (4, 4), "int32", "input", "cpu"),
        ("y", (4, 4, 4), "int32", "output", "cpu"),
    ]) as (x0, x1, y):
        with ir.For("i", 0, 4, nid="L1") as i:
            with ir.For("j", 0, 4, nid="L2") as j:
                with ir.VarDef("buf", (), "int32", "cache", "cpu") as b:
                    ir.MarkNid("S0")
                    b[()] = x0[i]
                    with ir.For("k", 0, 4, nid="L3") as k:
                        y[i, j, k] = b[()] * x1[i, j]
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    s.fission("L2", ir.FissionSide.After, "S0")
    ast = s.ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([
        ("x0", (4,), "int32", "input", "cpu"),
        ("x1", (4, 4), "int32", "input", "cpu"),
        ("y", (4, 4, 4), "int32", "output", "cpu"),
    ]) as (x0, x1, y):
        with ir.For("i", 0, 4) as i:
            with ir.VarDef("buf", (), "int32", "cache", "cpu") as b:
                b[()] = x0[i]
                with ir.For("j", 0, 4) as j:
                    with ir.For("k", 0, 4) as k:
                        y[i, j, k] = b[()] * x1[i, j]
    std = ir.pop_ast()

    assert std.match(ast)


def test_correct_dependency_overwritten_store():
    with ir.VarDef([
        ("x0", (4, 8), "int32", "input", "cpu"),
        ("x1", (4, 8), "int32", "input", "cpu"),
        ("y", (4, 8), "int32", "output", "cpu"),
    ]) as (x0, x1, y):
        with ir.For("i", 0, 4, nid="L1") as i:
            with ir.For("j", 0, 8, nid="L2") as j:
                with ir.VarDef("buf", (1,), "int32", "cache", "cpu") as b:
                    b[0] = 1  # (1)
                    ir.MarkNid("S0")
                    with ir.If(j > 1):
                        b[0] += x0[i, j] + x1[i, j]  # (2)
                    y[i, j] = b[0] * b[0]  # (3)
    # Explanation: (3)->(1) is a real dependency, while (3)->(2) is not.
    # We cannot determine b is loop-invarient just becase b[0] = 1
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    s.fission("L2", ir.FissionSide.After, "S0")
    ast = s.ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([
        ("x0", (4, 8), "int32", "input", "cpu"),
        ("x1", (4, 8), "int32", "input", "cpu"),
        ("y", (4, 8), "int32", "output", "cpu"),
    ]) as (x0, x1, y):
        with ir.For("i", 0, 4) as i:
            with ir.VarDef("buf", (8, 1), "int32", "cache", "cpu") as b:
                with ir.For("j", 0, 8) as j:
                    b[j, 0] = 1
                    with ir.If(j > 1):
                        b[j, 0] = x0[i, j] + x1[i, j] + 1
                with ir.For("j", 0, 8) as j:
                    y[i, j] = b[j, 0] * b[j, 0]
    std = ir.pop_ast()

    assert std.match(ast)
