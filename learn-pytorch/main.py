import torch
import numpy as np
import timeit

data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)
x_rand = torch.rand_like(x_data, dtype=torch.float)
print(f'{x_data.shape=}')
# torch. ones, zeros, rand, randn

x_big = torch.randn(10000, 10000)
y_big = torch.randn(10000, 10000)

number = 10
elapsed = timeit.timeit(lambda: x_big @ y_big, number=number)
print(f'x_big @ y_big CPU average time: {elapsed / number:.6f} seconds')

if torch.cuda.is_available():
    x_gpu = x_big.cuda()
    y_gpu = y_big.cuda()

    def gpu_matmul():
        result = x_gpu @ y_gpu
        torch.cuda.synchronize()
        return result

    gpu_matmul()
    elapsed_gpu = timeit.timeit(gpu_matmul, number=number)
    print(f'x_big @ y_big GPU average time: {elapsed_gpu / number:.6f} seconds')
else:
    print('CUDA is not available, skip GPU timing.')
