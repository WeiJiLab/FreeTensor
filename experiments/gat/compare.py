import sys
import torch
import numpy as np

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <dir1> <dir2>")
        exit(-1)

    dir1 = sys.argv[1]
    dir2 = sys.argv[2]

    #for name in ['y', 'd_x', 'd_w', 'd_w_attn_1', 'd_w_attn_2']:
    for name in ['y']:
        print(f"Comparing {name}")
        data1 = np.loadtxt(f"{dir1}/{name}.out")
        data2 = np.loadtxt(f"{dir2}/{name}.out")
        assert np.all(np.isclose(data2, data1)), f"{name} differs"
    print("All output matches")