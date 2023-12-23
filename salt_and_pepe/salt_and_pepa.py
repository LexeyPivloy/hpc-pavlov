import math
import random
import cupy as cp
import numpy as np
from PIL import Image
import time
import pandas as pd


def add_noise(img, prob):
    output = np.zeros(img.shape, np.uint8)
    thres = 1 - prob
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = img[i][j]
    return output


def img_arr_from_dir(img_directory, prob):
    img = np.array(Image.open(img_directory).convert("L"))

    if prob != 0:
        img_noise = add_noise(img, prob)
        img_out = Image.fromarray(img_noise).convert("L")
        img_out.save('adding_SnP_%d%s%d.bmp'%(img.shape[1], 'x', img.shape[0]))
        #img_out.show()
        return img_noise

    return img


def cpu_median(img, filter_size):
    index = filter_size // 2
    out = np.pad(img, pad_width=index, mode='edge')
    start = time.perf_counter()
    for i in range(index, len(img) - index):
        for j in range(index, len(img[0]) - index):
            out[i][j] = np.median(img[i - index:i + index + 1, j - index:j + index + 1])
    end = time.perf_counter()
    img_out = Image.fromarray(out[index:-index, index:-index]).convert("L")
    img_out.save('cpu_median_%s%d%s%d.bmp'%('cpu', img.shape[1], 'x', img.shape[0]))
    #img_out.show()
    return end - start


def cuda_median(img, filter_size):
    kernel = cp.RawKernel(r'''
        extern "C" 
        __global__ void m_filter(unsigned char* input, unsigned char* output, int width, int height) {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x < width && y < height) {
                unsigned char window[9];
                int index = 0;

                for (int i = -1; i <= 1; i++) {
                    for (int j = -1; j <= 1; j++) {
                        int nx = x + i;
                        int ny = y + j;

                        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                            //printf("%u", input[ny * width + nx]);
                            window[index] = input[ny * width + nx];
                        } else {
                            window[index] = 0;
                        }

                        index++;
                    }
                }

                for (int i = 0; i < 9; i++) {
                    for (int j = i + 1; j < 9; j++) {
                        if (window[i] > window[j]) {
                            unsigned char temp = window[i];
                            window[i] = window[j];
                            window[j] = temp;
                        }
                    }
                }
                output[y * width + x] = window[4];
            }
        }
        ''',
                          'm_filter')
    index = filter_size // 2
    img_new = np.pad(img, pad_width=index, mode='edge')
    img_arr_gpu = cp.asarray(img_new.flatten())
    result_gpu = cp.zeros(img_arr_gpu.shape[0], dtype=cp.uint8)

    threadsperblock = (16, 16)
    blockspergrid_x = int(np.ceil(img.shape[0] / threadsperblock[0]))
    blockspergrid_y = int(np.ceil(img.shape[1] / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    warp_size = 32
    blockspergrid = (math.ceil(blockspergrid[0] / warp_size) * warp_size,
                     math.ceil(blockspergrid[1] / warp_size) * warp_size)

    start = time.perf_counter()
    kernel(blockspergrid, threadsperblock, (img_arr_gpu, result_gpu, img_new.shape[1], img_new.shape[0]))
    end = time.perf_counter()

    result_gpu = np.array(result_gpu.get()).reshape(len(img_new), len(img_new[0]))

    img_out = Image.fromarray(result_gpu[index:-index, index:-index]).convert("L")
    img_out.save('gpu_median_%s%d%s%d.bmp'%('gpu', img.shape[1], 'x', img.shape[0]))
    #img_out.show()
    return end - start


if __name__ == '__main__':

    noise_prob = 0.02

    images_wout_noise = ["wout0.bmp", "wout1.bmp", "wout2.bmp"]
    images_w_noise = ["w0.bmp", "w0.bmp", "w1.bmp", "w2.bmp"]
    res_list = []
    for image in images_w_noise:
        img_arr = img_arr_from_dir(image, 0)
        res_dict = {"resolution": (img_arr.shape[1],img_arr.shape[0]),
                    "adding SnP": 0}

        res_dict["gpu"] = cuda_median(img_arr, 3)
        res_dict["cpu"] = cpu_median(img_arr, 3)

        res_dict["acc (gpu>cpu)"] = res_dict["cpu"] / res_dict["gpu"]

        res_list.append(res_dict)

    for image in images_wout_noise:
        img_arr = img_arr_from_dir(image, noise_prob)
        res_dict = {"resolution": (img_arr.shape[1],img_arr.shape[0]),
                    "adding SnP": noise_prob}

        res_dict["gpu"] = cuda_median(img_arr, 3)
        res_dict["cpu"] = cpu_median(img_arr, 3)

        res_dict["acc (gpu>cpu)"] = res_dict["cpu"] / res_dict["gpu"]

        res_list.append(res_dict)

    df1 = pd.DataFrame.from_records(res_list)
    print(df1.to_markdown(index=False))
