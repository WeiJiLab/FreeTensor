import freetensor as ft
import pytest


def test_basic():
    with ft.VarDef("x", (4, 4), "float32", "output", "cpu") as x:
        x[2, 3] = 2.0
        x[1, 0] = 3.0
    ast = ft.pop_ast()
    txt = ft.dump_ast(ast)
    print(txt)
    ast2 = ft.load_ast(txt)
    print(ast2)
    assert ast2.match(ast)


def test_func():
    with ft.VarDef("x", (4, 4), "float32", "output", "cpu") as x:
        x[2, 3] = 2.0
        x[1, 0] = 3.0
    func = ft.lower(ft.Func("main", ["x"], [], ft.pop_ast()), ft.CPU())
    txt = ft.dump_ast(func)
    print(txt)
    func2 = ft.load_ast(txt)
    print(func2)
    assert func2.body.match(func.body)
    assert func2.name == "main"


def test_func_with_return_value():
    with ft.VarDef("x", (4, 4), "float32", "output", "cpu") as x:
        x[2, 3] = 2.0
        x[1, 0] = 3.0
    func = ft.lower(
        ft.Func("main", [], [("x", ft.DataType("float32"))], ft.pop_ast()),
        ft.CPU())
    txt = ft.dump_ast(func)
    print(txt)
    func2 = ft.load_ast(txt)
    print(func2)
    assert func2.body.match(func.body)
    assert func2.name == "main"


def test_scalar_op():
    with ft.VarDef([("x", (), "int32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        y[()] = x[()] * 2 + 1
    ast = ft.pop_ast()
    txt = ft.dump_ast(ast)
    print(txt)
    ast2 = ft.load_ast(txt)
    print(ast2)
    assert ast2.match(ast)


def test_cast():
    with ft.VarDef([("x", (), "float32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        y[()] = ft.cast(x[()], "int32") * 2
    ast = ft.pop_ast()
    txt = ft.dump_ast(ast)
    print(txt)
    ast2 = ft.load_ast(txt)
    print(ast2)
    assert ast2.match(ast)


def test_intrinsic():
    with ft.VarDef([("x", (), "float32", "input", "cpu"),
                    ("y", (), "float32", "output", "cpu")]) as (x, y):
        y[()] = ft.intrinsic("sinf(%)", x[()], ret_type="float32")
    ast = ft.pop_ast()
    txt = ft.dump_ast(ast)
    print(txt)
    ast2 = ft.load_ast(txt)
    print(ast2)
    assert ast2.match(ast)


def test_intrinsic_side_effect():
    with ft.VarDef([("x1", (), "float32", "input", "cpu"),
                    ("x2", (), "float32", "input", "cpu")]) as (x1, x2):
        ft.Eval(ft.intrinsic("foo(%, %)", x1[()], x2[()], has_side_effect=True))
    ast = ft.pop_ast()
    txt = ft.dump_ast(ast)
    print(txt)
    ast2 = ft.load_ast(txt)
    print(ast2)
    assert ast2.match(ast)


def test_for():
    with ft.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 4) as i:
            y[i] = x[i] + 1
    ast = ft.pop_ast()
    txt = ft.dump_ast(ast)
    print(txt)
    ast2 = ft.load_ast(txt)
    print(ast2)
    assert ast2.match(ast)


def test_reversed_for():
    with ft.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i", 3, -1, -1) as i:
            y[i] = x[i] + 1
    ast = ft.pop_ast()
    txt = ft.dump_ast(ast)
    print(txt)
    ast2 = ft.load_ast(txt)
    print(ast2)
    assert ast2.match(ast)


def test_for_with_multiple_properties():
    with ft.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 4, nid="foo", no_deps=['x'], prefer_libs=True) as i:
            y[i] = x[i] + 1
    ast = ft.pop_ast()
    txt = ft.dump_ast(ast)
    print(txt)
    ast2 = ft.load_ast(txt)
    print(ast2)
    assert ast2.match(ast)
    s = ft.Schedule(ast2)
    assert s.find("foo").property.no_deps == ['x']
    assert s.find("foo").property.prefer_libs


def test_for_with_parallel():
    with ft.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 4, nid="foo") as i:
            y[i] = x[i] + 1
    s = ft.Schedule(ft.pop_ast())
    s.parallelize("foo", "openmp")
    ast = s.ast()
    txt = ft.dump_ast(ast)
    print(txt)
    ast2 = ft.load_ast(txt)
    print(ast2)
    assert ast2.match(ast)
    s = ft.Schedule(ast2)
    assert s.find("foo").property.parallel == ft.ffi.ParallelScope("openmp")


def test_for_with_parallel_reduction():
    with ft.VarDef([("x", (4, 64), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "inout", "cpu")]) as (x, y):
        with ft.For("i", 0, 4, nid="L1") as i:
            with ft.For("j", 0, 64, nid="L2") as j:
                y[i] = y[i] + x[i, j]
    s = ft.Schedule(ft.pop_ast())
    s.parallelize("L2", "openmp")
    ast = ft.lower(s.ast(), skip_passes=["cpu_lower_parallel_reduction"])
    txt = ft.dump_ast(ast)
    print(txt)
    ast2 = ft.load_ast(txt)
    print(ast2)
    assert ast2.match(ast)
    s = ft.Schedule(ast2)
    assert s.find("L2").property.parallel == ft.ffi.ParallelScope("openmp")
    assert s.find("L2").property.reductions[0].var == "y"


def test_if():
    with ft.VarDef("y", (4,), "int32", "output", "cpu") as y:
        with ft.For("i", 0, 4) as i:
            with ft.If(i < 2):
                y[i] = 0
            with ft.Else():
                y[i] = 1
    ast = ft.pop_ast()
    txt = ft.dump_ast(ast)
    print(txt)
    ast2 = ft.load_ast(txt)
    print(ast2)
    assert ast2.match(ast)


def test_assert():
    with ft.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 4) as i:
            with ft.Assert(x[i] != 0):
                y[i] = 100 // x[i]
    ast = ft.pop_ast()
    txt = ft.dump_ast(ast)
    print(txt)
    ast2 = ft.load_ast(txt)
    print(ast2)
    assert ast2.match(ast)


def test_id():
    with ft.VarDef("x", (4, 4), "float32", "output", "cpu") as x:
        ft.MarkNid("foo")
        x[2, 3] = 2.0
        x[1, 0] = 3.0
    ast = ft.pop_ast()
    txt = ft.dump_ast(ast)
    print(txt)
    ast2 = ft.load_ast(txt)
    print(ast2)
    assert ast2.match(ast)
    s = ft.Schedule(ast2)
    assert s.find("foo").type() == ft.ASTNodeType.Store


def test_id_of_stmt_seq():
    with ft.VarDef("x", (4, 4), "float32", "output", "cpu") as x:
        with ft.NamedScope("foo"):
            x[2, 3] = 2.0
            x[1, 0] = 3.0
    ast = ft.pop_ast()
    txt = ft.dump_ast(ast)
    print(txt)
    ast2 = ft.load_ast(txt)
    print(ast2)
    assert ast2.match(ast)
    s = ft.Schedule(ast2)
    assert s.find("foo").type() == ft.ASTNodeType.StmtSeq


def test_empty_stmt_seq():
    with ft.VarDef("x", (4, 4), "float32", "output", "cpu") as x:
        x[2, 3] = 2.0
        x[1, 0] = 3.0
        with ft.NamedScope("foo"):
            pass
    ast = ft.pop_ast()
    txt = ft.dump_ast(ast)
    print(txt)
    ast2 = ft.load_ast(txt)
    print(ast2)
    assert ast2.match(ast)


def test_complex_name():
    with ft.VarDef("x!@#$%^&*", (4, 4), "float32", "output", "cpu") as x:
        x[2, 3] = 2.0
        x[1, 0] = 3.0
    ast = ft.pop_ast()
    txt = ft.dump_ast(ast)
    print(txt)
    ast2 = ft.load_ast(txt)
    print(ast2)
    assert ast2.match(ast)


def test_complex_id():
    with ft.VarDef("x", (4, 4), "float32", "output", "cpu") as x:
        ft.MarkNid("id!@#$%^&*")
        x[2, 3] = 2.0
        x[1, 0] = 3.0
    ast = ft.pop_ast()
    txt = ft.dump_ast(ast)
    print(txt)
    ast2 = ft.load_ast(txt)
    print(ast2)
    assert ast2.match(ast)
    s = ft.Schedule(ast2)
    assert s.find("id!@#$%^&*").type() == ft.ASTNodeType.Store


def test_var_name_be_same_with_builtin():
    with ft.VarDef([("x1", (4,), "int32", "input", "cpu"),
                    ("x2", (4,), "int32", "input", "cpu"),
                    ("max", (4,), "int32", "output", "cpu")]) as (x1, x2, y):
        with ft.For("i", 0, 4) as i:
            y[i] = ft.max(x1[i], x2[i])
    ast = ft.pop_ast()
    txt = ft.dump_ast(ast)
    print(txt)
    ast2 = ft.load_ast(txt)
    print(ast2)
    assert ast2.match(ast)


def test_var_name_be_same_with_keyword():
    with ft.VarDef([("x1", (4,), "int32", "input", "cpu"),
                    ("x2", (4,), "int32", "input", "cpu"),
                    ("if", (4,), "int32", "output", "cpu")]) as (x1, x2, y):
        with ft.For("i", 0, 4) as i:
            y[i] = ft.if_then_else(x1[i] < x2[i], -1, 1)
    ast = ft.pop_ast()
    txt = ft.dump_ast(ast)
    print(txt)
    ast2 = ft.load_ast(txt)
    print(ast2)
    assert ast2.match(ast)
