
<div>
    <h1>Удаление шума "Соль и перец" на изображениях</h1>
    <p>В таблице ниже приведены значения замеров времени для медианного фильтра в зависимости от разрешения изображений. Было выбрано 6 изображений (3 с изначально добавленным шумом взяты из интернета и 3 изображения, на которые применялся написанный код "соли и перца". Добавленный шум приведен во втором столбце). Написана реализация ядра на C++ (столбец в таблице: gpu) и время работы без распаралеливания (столбец в таблице: cpu)</p>
    <p>Последний столбец показывает ускорение работы программы с использованием ядра на C++ относительно однопоточной реализации (acc (gpu>cpu))</p>
</div>


| resolution   |   adding SnP |       gpu |       cpu |   acc (gpu>cpu) |
|:-------------|-------------:|----------:|----------:|----------------:|
| (240, 150)   |         0    | 7.86e-05  |  0.808057 |      10280.6    |
| (320, 428)   |         0    | 3.52e-05  |  3.21069  |      91212.7    |
| (512, 512)   |         0    | 4.21e-05  |  5.9033   |     140221      |
| (300, 224)   |         0.02 | 3.81e-05  |  1.54528  |      40558.4    |
| (700, 700)   |         0.02 | 3.92e-05  | 11.1136   |     283511      |
| (2048, 1534) |         0.02 | 0.0001056 | 70.04     |     663258      |


<div>
    <h2>Таблица изображений</h2>
</div>

| resolution   | input                                                                                  | add noise                                                                                              | cpu                                                                                                       | gpu                                                                                                       |
|--------------|----------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|
| (240, 150)   | ![](https://github.com/LexeyPivloy/hpc-pavlov/blob/main/salt-and-pepe/input/w0.bmp)    |                                                                                                        | ![](https://github.com/LexeyPivloy/hpc-pavlov/blob/main/salt-and-pepe/output/cpu_median_cpu240x150.bmp)   | ![](https://github.com/LexeyPivloy/hpc-pavlov/blob/main/salt-and-pepe/output/cpu_median_gpu240x150.bmp)   |
| (320, 428)   | ![](https://github.com/LexeyPivloy/hpc-pavlov/blob/main/salt-and-pepe/input/w1.bmp)    |                                                                                                        | ![](https://github.com/LexeyPivloy/hpc-pavlov/blob/main/salt-and-pepe/output/cpu_median_cpu320x428.bmp)   | ![](https://github.com/LexeyPivloy/hpc-pavlov/blob/main/salt-and-pepe/output/cpu_median_gpu320x428.bmp)   |
| (512, 512)   | ![](https://github.com/LexeyPivloy/hpc-pavlov/blob/main/salt-and-pepe/input/w2.bmp)    |                                                                                                        | ![](https://github.com/LexeyPivloy/hpc-pavlov/blob/main/salt-and-pepe/output/cpu_median_cpu515x512.bmp)   | ![](https://github.com/LexeyPivloy/hpc-pavlov/blob/main/salt-and-pepe/output/cpu_median_gpu515x512.bmp)   |
| (300, 224)   | ![](https://github.com/LexeyPivloy/hpc-pavlov/blob/main/salt-and-pepe/input/wout0.bmp) | ![](https://github.com/LexeyPivloy/hpc-pavlov/blob/main/salt-and-pepe/output/adding_SnP_300x224.bmp)   | ![](https://github.com/LexeyPivloy/hpc-pavlov/blob/main/salt-and-pepe/output/cpu_median_gpu300x224.bmp)   | ![](https://github.com/LexeyPivloy/hpc-pavlov/blob/main/salt-and-pepe/output/cpu_median_cpu300x224.bmp)   |
| (700, 700)   | ![](https://github.com/LexeyPivloy/hpc-pavlov/blob/main/salt-and-pepe/input/wout1.bmp) | ![](https://github.com/LexeyPivloy/hpc-pavlov/blob/main/salt-and-pepe/output/adding_SnP_700x700.bmp)   | ![](https://github.com/LexeyPivloy/hpc-pavlov/blob/main/salt-and-pepe/output/cpu_median_gpu700x700.bmp)   | ![](https://github.com/LexeyPivloy/hpc-pavlov/blob/main/salt-and-pepe/output/cpu_median_cpu700x700.bmp)   |
| (2048, 1534) | ![](https://github.com/LexeyPivloy/hpc-pavlov/blob/main/salt-and-pepe/input/wout2.bmp) | ![](https://github.com/LexeyPivloy/hpc-pavlov/blob/main/salt-and-pepe/output/adding_SnP_2048x1534.bmp) | ![](https://github.com/LexeyPivloy/hpc-pavlov/blob/main/salt-and-pepe/output/cpu_median_gpu2048x1534.bmp) | ![](https://github.com/LexeyPivloy/hpc-pavlov/blob/main/salt-and-pepe/output/cpu_median_cpu2048x1534.bmp) |


<h3><i>Вывод</i>: распараллеливание на gpu отлично справляется с поставленной задачей </h3>

![](https://github.com/LexeyPivloy/hpc-pavlov/blob/main/static/nerd_SnP.gif)