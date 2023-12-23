import math
import random
import cupy as cp
import numpy as np
from PIL import Image
import time
import pandas as pd
from numba import cuda, uint8


def img_arr_from_dir(img_directory):
    img_arr = np.array(Image.open(img_directory).convert("L"))
    print(img_arr.shape)
    return img_arr


@cuda.jit
def gpu_kernel(img, result):
    i, j = cuda.grid(2)

    if i < result.shape[0] and j < result.shape[1]:
        x = i // 2
        y = j // 2
        if x < img.shape[0] - 1 and y < img.shape[1] - 1:
            fx = i % 2
            fy = j % 2
            fx1 = min(x + 1, img.shape[0] - 1)
            fy1 = min(y + 1, img.shape[1] - 1)

            result[i, j] = (
                img[x, y] * (1 - fx) * (1 - fy) +
                img[x, fy1] * (1 - fx) * fy +
                img[fx1, y] * fx * (1 - fy) +
                img[fx1, fy1] * fx * fy
            )


def gpu_bilinear(img):

    threadsperblock = (8, 8)
    blockspergrid = (math.ceil(img.shape[0] * 2 / threadsperblock[0]), math.ceil(img.shape[1] * 2 / threadsperblock[1]))
    start = cuda.event()
    end = cuda.event()

    img_arr_gpu = cuda.to_device(img)
    result_gpu = cuda.to_device(cp.zeros((img.shape[0] * 2, img.shape[1] * 2), dtype=cp.uint8))

    start = time.perf_counter()
    gpu_kernel[blockspergrid, threadsperblock](img_arr_gpu, result_gpu)
    end = time.perf_counter()

    result = result_gpu.copy_to_host()
    img_gpu = Image.fromarray(result.reshape((img.shape[0] * 2, img.shape[1] * 2)))
    img_gpu = img_gpu.convert("L")
    img_gpu.save('bileniar_%s%d%s%d.bmp'%('gpu', img.shape[1], 'x', img.shape[0]))
    #img_gpu.show()

    return end - start


def cpu_bilinear(img):
    result = np.zeros((img.shape[0] * 2, img.shape[1] * 2), dtype=np.float32)

    start = time.perf_counter()
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            x = i // 2
            y = j // 2
            if x < img.shape[0] - 1 and y < img.shape[1] - 1:
                fx = i % 2
                fy = j % 2
                fx1 = min(x + 1, img.shape[0] - 1)
                fy1 = min(y + 1, img.shape[1] - 1)

                result[i, j] = (
                    img[x, y] * (1 - fx) * (1 - fy) +
                    img[x, fy1] * (1 - fx) * fy +
                    img[fx1, y] * fx * (1 - fy) +
                    img[fx1, fy1] * fx * fy
                )

    end = time.perf_counter()

    img_gpu = Image.fromarray(result.reshape((img.shape[0] * 2, img.shape[1] * 2)))
    img_gpu = img_gpu.convert("L")
    img_gpu.save('bileniar_%s%d%s%d.bmp'%('cpu', img.shape[1], 'x', img.shape[0]))
    #img_gpu.show()
    return end - start



if __name__ == '__main__':
    images = ["100.bmp", "100.bmp", "200.bmp", "240.bmp", "300.bmp", "428.bmp", "600.bmp"]
    res_list = []
    for image in images:
        img_arr = img_arr_from_dir(image)
        res_dict = {"resolution inp": (img_arr.shape[1], img_arr.shape[0])}

        res_dict["cpu"] = cpu_bilinear(img_arr)
        res_dict["gpu"] = gpu_bilinear(img_arr)

        res_dict["acc (gpu>cpu)"] = res_dict["cpu"] / res_dict["gpu"]

        res_list.append(res_dict)

    df1 = pd.DataFrame.from_records(res_list)
    print(df1.to_markdown(index=False))

