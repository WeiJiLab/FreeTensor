import freetensor
import pytest


def test_vector_add():
    # Used in README.md and docs/guide/schedules.md

    import freetensor as ft
    import numpy as np

    n = 4

    # Change this line to ft.optimize(verbose=1) to see the resulting native code
    @ft.optimize
    def test(a: ft.Var[(n,), "int32"], b: ft.Var[(4,), "int32"]):
        y = ft.empty((n,), "int32")
        for i in range(n):
            y[i] = a[i] + b[i]
        return y

    y = test(np.array([1, 2, 3, 4], dtype="int32"),
             np.array([2, 3, 4, 5], dtype="int32")).numpy()
    print(y)

    assert np.array_equal(y, [3, 5, 7, 9])


def test_vector_add_dynamic_length():
    # Used in README.md and docs/guide/schedules.md

    import freetensor as ft
    import numpy as np

    @ft.optimize
    def test(n: ft.Var[(), "int32"], a, b):
        a: ft.Var[(n,), "int32"]
        b: ft.Var[(n,), "int32"]
        y = ft.empty((n,), "int32")
        for i in range(n):
            y[i] = a[i] + b[i]
        return y

    y = test(np.array(4, dtype="int32"), np.array([1, 2, 3, 4], dtype="int32"),
             np.array([2, 3, 4, 5], dtype="int32")).numpy()
    print(y)

    assert np.array_equal(y, [3, 5, 7, 9])


@pytest.mark.skipif(not freetensor.with_cuda(), reason="requires CUDA")
def test_vector_add_gpu():
    # Used in README.md

    import freetensor as ft
    import numpy as np

    # Using the 0-th GPU device
    with ft.Device(ft.GPU(), 0):

        @ft.optimize(
            # Parallel Loop Li as GPU threads
            schedule_callback=lambda s: s.parallelize("Li", "threadIdx.x"))
        # Use "byvalue" for `n` so it can be used both during kernel launching
        # and inside a kernel
        def test(n: ft.Var[(), "int32", "input", "byvalue"], a, b):
            a: ft.Var[(n,), "int32"]
            b: ft.Var[(n,), "int32"]
            y = ft.empty((n,), "int32")
            #! nid: Li # Name the loop below as "Li"
            for i in range(n):
                y[i] = a[i] + b[i]
            return y

        y = test(np.array(4, dtype="int32"),
                 np.array([1, 2, 3, 4], dtype="int32"),
                 np.array([2, 3, 4, 5], dtype="int32")).numpy()
        print(y)

    assert np.array_equal(y, [3, 5, 7, 9])


def test_vector_add_libop():
    # Used in README.md

    import freetensor as ft
    import numpy as np

    @ft.optimize
    def test(n: ft.Var[(), "int32"], a, b):
        a: ft.Var[(n,), "int32"]
        b: ft.Var[(n,), "int32"]
        y = a + b  # Or y = ft.add(a, b)
        return y

    y = test(np.array(4, dtype="int32"), np.array([1, 2, 3, 4], dtype="int32"),
             np.array([2, 3, 4, 5], dtype="int32")).numpy()
    print(y)

    assert np.array_equal(y, [3, 5, 7, 9])


def test_dynamic_and_static():
    # Used in docs/guide/first-program.md

    import freetensor as ft
    import numpy as np

    n = 4

    @ft.optimize
    def test(a: ft.Var[(n,), "int32"], b: ft.Var[(4,), "int32"],
             c: ft.Var[(4,), "int32"]):
        inputs = [a, b, c]  # Static
        y = ft.empty((n,), "int32")  # Dynamic
        for i in range(n):  # Dyanmic
            y[i] = 0  # Dynamic
            for item in inputs:  # Static
                y[i] += item[i]  # Dynamic
        return y

    y = test(np.array([1, 2, 3, 4], dtype="int32"),
             np.array([2, 3, 4, 5], dtype="int32"),
             np.array([3, 4, 5, 6], dtype="int32")).numpy()
    print(y)

    assert np.array_equal(y, [6, 9, 12, 15])


def test_parallel_vector_add():
    # Used in docs/guide/schedules.md

    import freetensor as ft
    import numpy as np

    n = 4

    # Add verbose=1 to see the resulting native code
    @ft.optimize(schedule_callback=lambda s: s.parallelize('Li', 'openmp')
                )  # <-- 2. Apply the schedule
    def test(a: ft.Var[(n,), "int32"], b: ft.Var[(4,), "int32"]):
        y = ft.empty((n,), "int32")
        #! nid: Li  # <-- 1. Name the loop as Li
        for i in range(n):
            y[i] = a[i] + b[i]
        return y

    y = test(np.array([1, 2, 3, 4], dtype="int32"),
             np.array([2, 3, 4, 5], dtype="int32")).numpy()
    print(y)

    assert np.array_equal(y, [3, 5, 7, 9])


def test_split_and_parallel_vector_add():
    # Used in docs/guide/schedules.md

    import freetensor as ft
    import numpy as np

    n = 1024

    def sch(s):
        outer, inner = s.split('Li', 32)
        s.parallelize(outer, 'openmp')

    # Set verbose=1 to see the resulting native code
    # Set verbose=2 to see the code after EVERY schedule
    @ft.optimize(schedule_callback=sch)
    def test(a: ft.Var[(n,), "int32"], b: ft.Var[(4,), "int32"]):
        y = ft.empty((n,), "int32")
        #! nid: Li
        for i in range(n):
            y[i] = a[i] + b[i]
        return y

    y = test(np.array(np.arange(1024), dtype="int32"),
             np.array(np.arange(1024), dtype="int32")).numpy()
    print(y)

    assert np.array_equal(y, np.arange(0, 2048, 2))
