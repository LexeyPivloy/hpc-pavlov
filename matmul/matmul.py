import numpy as np
import cupy as cp
import time
from numba import cuda
import math
import pandas as pd


file = open('test.txt', 'w')

def get_cpu_matmul_result(a, b, n):
    c = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            c[i, j] = 0
            for k in range(n):
                c[i, j] += a[i, j]*b[i, j]
    return c

def numpy_matmul(size):
    x_cpu = np.linspace(1, size ** 2, size ** 2).reshape(size, size)
    y_cpu = np.linspace(1, size ** 2, size ** 2).reshape(size, size)
    start_time = time.perf_counter()
    ans_cpu = np.matmul(x_cpu, y_cpu)
    end_time = time.perf_counter()
    file.write(str(end_time - start_time) + ' ' + str(size) + " cpu np\n")
    return end_time - start_time

def cpu_matmul(size):
    x_cpu = np.linspace(1, size ** 2, size ** 2).reshape(size, size)
    y_cpu = np.linspace(1, size ** 2, size ** 2).reshape(size, size)
    start_time = time.perf_counter()
    ans = get_cpu_matmul_result(x_cpu, y_cpu, size)
    end_time = time.perf_counter()
    file.write(str(end_time - start_time) + ' ' + str(size) + " cpu\n")
    return end_time - start_time

def cupy_matmul(size):
    x_gpu = cp.linspace(1, size ** 2, size ** 2).reshape(size, size)
    y_gpu = cp.linspace(1, size ** 2, size ** 2).reshape(size, size)
    start_time = time.perf_counter()
    ans_gpu = cp.matmul(x_gpu, y_gpu)
    end_time = time.perf_counter()
    file.write(str(end_time - start_time) + ' ' + str(size) + " gpu cp\n")
    return end_time - start_time

@cuda.jit
def get_numba_matmul_result(A, B, C):
    row, col = cuda.grid(2)
    if row < C.shape[0] and col < C.shape[1]:
        tmp = 0.
        for k in range(A.shape[1]):
            tmp += A[row, k] * B[k, col]
        C[row, col] = tmp


def numba_matmul(size):

    x_numba = cp.linspace(1, size ** 2, size ** 2).reshape(size, size)
    y_numba = cp.linspace(1, size ** 2, size ** 2).reshape(size, size)
    c_numba = cp.zeros((x_numba.shape[0], y_numba.shape[1]), dtype=np.float32)

    d_X = cuda.to_device(x_numba)
    d_Y = cuda.to_device(y_numba)
    d_C = cuda.device_array_like(c_numba)

    threadsperblock = (16, 16)
    blockspergrid_x = int(np.ceil(x_numba.shape[0] / threadsperblock[0]))
    blockspergrid_y = int(np.ceil(x_numba.shape[1] / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    warp_size = 32
    blockspergrid = (math.ceil(blockspergrid[0] / warp_size) * warp_size,
                     math.ceil(blockspergrid[1] / warp_size) * warp_size)

    start_time = time.perf_counter()
    get_numba_matmul_result[blockspergrid, threadsperblock](d_X, d_Y, d_C)
    end_time = time.perf_counter()
    file.write(str(end_time - start_time) + ' ' + str(size) + " gpu numba\n")
    return end_time - start_time


add_kernel = cp.RawKernel(r'''
extern "C" __global__
void matmul(const float* a, const float* b, float* c, int width) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    float value = 0.0, el1 = 0.0, el2 = 0.0;

    for(int i = 0; i < width; i++)
	{
		el1 = a[y * width + i];
		el2 = b[i * width + x];

		value += el1 * el2;
	}

    c[y * width + x] = value;
}
''',
"matmul")

def gpu_c_matmul(size):
    x_cpu = cp.linspace(1, size ** 2, size ** 2).reshape(size, size)
    y_cpu = cp.linspace(1, size ** 2, size ** 2).reshape(size, size)
    c_cpu = cp.zeros((size, size), dtype=cp.float32)

    threadsperblock = (16, 16)
    blockspergrid_x = int(np.ceil(x_cpu.shape[0] / threadsperblock[0]))
    blockspergrid_y = int(np.ceil(x_cpu.shape[1] / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    warp_size = 32
    blockspergrid = (math.ceil(blockspergrid[0] / warp_size) * warp_size,
                     math.ceil(blockspergrid[1] / warp_size) * warp_size)

    start_time = time.perf_counter()
    res = add_kernel(blockspergrid, threadsperblock, (x_cpu, y_cpu, c_cpu, size))
    end_time = time.perf_counter()
    file.write(str(end_time - start_time) + ' ' + str(size) + " gpu C\n")
    return end_time - start_time

sizes = [50, 100, 200, 400, 800, 1600, 2000]
res_list = []




for size in sizes:
    res_dict = {"size": size}

    res_dict["numpy"] = numpy_matmul(size)
    res_dict["cpu"] = cpu_matmul(size) if (size <= 800) else "too boring"
    res_dict["cupy"] = cupy_matmul(size)
    res_dict["numba"] = numba_matmul(size)
    res_dict["gpu"] = gpu_c_matmul(size)

    res_dict["acc (gpu>cpu)"] = res_dict["cpu"] / res_dict["gpu"] if (size <= 800) else "too much"
    res_dict["acc (gpu>numpy)"] = res_dict["numpy"] / res_dict["gpu"]

    res_list.append(res_dict)

    file.write("\n")

df1 = pd.DataFrame.from_records(res_list)
print(df1.to_markdown(index = False))

file.close()