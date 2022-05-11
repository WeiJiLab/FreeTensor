import freetensor as ft
import pytest
import inspect


# DO NOT ADD LINES IN THIS FUNCTION
def test_illegal_bin_op():

    @ft.inline
    def bar(a, b):
        a @ b

    @ft.inline
    def foo(a, b):
        bar(a, b)

    with pytest.raises(ft.StagingError) as e:

        @ft.transform
        def test(a, b):
            a: ft.Var[(1,), "int32", "input", "cpu"]
            b: ft.Var[(1,), "int32", "input", "cpu"]
            foo(a, b)

    frame_info = inspect.getframeinfo(inspect.currentframe())
    line_foo = frame_info.lineno - 2
    line_bar = frame_info.lineno - 10
    line_ab = frame_info.lineno - 14
    file = frame_info.filename

    print(e.value.args[0])
    assert f"File \"{file}\", line {line_foo}" in e.value.args[0]
    assert f"File \"{file}\", line {line_bar}" in e.value.args[0]
    assert f"File \"{file}\", line {line_ab}" in e.value.args[0]
