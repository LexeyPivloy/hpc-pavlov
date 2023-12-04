import numpy as np
import cupy as cp
import time
from numba import cuda
import math
import pandas as pd


file = open('testVecSum.txt', 'w')

def get_cpu_vecsum_result(n):
    sum = 0
    for i in range(len(n)):
        sum += n[i]
    return sum

def numpy_vecsum(size):
    x = np.random.random(size)
    start_time = time.perf_counter()
    ans_cpu = np.sum(x)
    end_time = time.perf_counter()
    file.write(str(end_time - start_time) + ' ' + str(size) + " cpu np\n")
    return end_time - start_time

def cpu_vecsum(size):
    x = np.random.random(size)
    start_time = time.perf_counter()
    ans = get_cpu_vecsum_result(x)
    end_time = time.perf_counter()
    file.write(str(end_time - start_time) + ' ' + str(size) + " cpu\n")
    return end_time - start_time

def cupy_vecsum(size):
    x = cp.random.random(size)
    start_time = time.perf_counter()
    ans_gpu = cp.sum(x)
    end_time = time.perf_counter()
    file.write(str(end_time - start_time) + ' ' + str(size) + " gpu cp\n")
    return end_time - start_time

@cuda.jit
def get_numba_vecsum_result(a, c):
    i = cuda.grid(1)
    if i < c.size:
        c[i] = a[i]



def numba_vecsum(size):

    x_numba = cp.random.random(size)
    c_numba = cp.empty_like(x_numba)

    d_X = cuda.to_device(x_numba)
    d_C = cuda.device_array_like(c_numba)

    threadsperblock = 32
    blockspergrid = (c_numba.size + (threadsperblock - 1)) // threadsperblock

    start_time = time.perf_counter()
    get_numba_vecsum_result[blockspergrid, threadsperblock](d_X, d_C)
    end_time = time.perf_counter()
    file.write(str(end_time - start_time) + ' ' + str(size) + " gpu numba\n")
    return end_time - start_time


add_kernel = cp.RawKernel(r'''
extern "C" __global__
void vec_sum(const int* a, int* b, const int size) {
    int gridSize = blockDim.x * gridDim.x;
    int first_index = blockIdx.x * blockDim.x + threadIdx.x;
    for (int index = first_index; index < size; index += gridSize)
    {
        atomicAdd(&b[0], a[index]);
    }
}
''', 'vec_sum')

def gpu_c_vecsum(size):
    x = cp.random.random(size)
    c = cp.empty_like(x)

    threadsperblock = (32,)
    blockspergrid = ((x.size + (threadsperblock[0] - 1)) // threadsperblock[0], )

    start_time = time.perf_counter()
    res = add_kernel(blockspergrid, threadsperblock, (x, c, size))
    end_time = time.perf_counter()
    file.write(str(end_time - start_time) + ' ' + str(size) + " gpu C\n")
    return end_time - start_time

sizes = [10, 1000, 5000, 20000, 100000, 400000, 700000, 1000000]
res_list = []




for size in sizes:
    res_dict = {"size": size}

    res_dict["numpy"] = numpy_vecsum(size)
    res_dict["cpu"] = cpu_vecsum(size)
    res_dict["cupy"] = cupy_vecsum(size)
    res_dict["numba"] = numba_vecsum(size)
    res_dict["gpu"] = gpu_c_vecsum(size)

    res_dict["acc (gpu>cpu)"] = res_dict["cpu"] / res_dict["gpu"]
    res_dict["acc (gpu>numpy)"] = res_dict["numpy"] / res_dict["gpu"]

    res_list.append(res_dict)

    file.write("\n")

df1 = pd.DataFrame.from_records(res_list)
print(df1.to_markdown(index = False))

file.close()