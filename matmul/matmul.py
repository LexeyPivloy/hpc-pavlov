import numpy as np
import cupy as cp
import time

file = open('test.txt', 'w')

def cpu_matmul(a, b, n):
    c = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            c[i, j] = 0
            for k in range(n):
                c[i, j] += a[i, j]*b[i, j]
    return c


sizes = [50, 100, 200, 400, 800, 1600, 2000]
for i in sizes:
    x_cpu = np.linspace(1, i ** 2, i ** 2).reshape(i, i)
    y_cpu = np.linspace(1, i ** 2, i ** 2).reshape(i, i)
    start_time = time.perf_counter()
    ans_cpu = np.matmul(x_cpu, y_cpu)
    end_time = time.perf_counter()
    file.write(str(end_time - start_time) + ' ' + str(i) + " cpu np\n")
    #start_time = time.perf_counter()
    #ans = cpu_matmul(x_cpu, y_cpu, i)
    #end_time = time.perf_counter()
    #file.write(str(end_time - start_time) + ' ' + str(i) + " cpu\n")
    x_gpu = cp.linspace(1, i ** 2, i ** 2).reshape(i, i)
    y_gpu = cp.linspace(1, i ** 2, i ** 2).reshape(i, i)
    start_time = time.perf_counter()
    ans_gpu = cp.matmul(x_gpu, y_gpu)
    end_time = time.perf_counter()
    file.write(str(end_time - start_time) + ' ' + str(i) + " gpu\n")


file.close()