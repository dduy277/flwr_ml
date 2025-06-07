import sys
import torch

print(sys.executable)
print(torch.device("xpu:0"))
torch.xpu.get_device_name(0)
# print(torch.xpu.get_device_name(0))
# print(torch.xpu.get_device_capability(0))

# import ray
# import time

# ray.init(num_gpus=1)

# @ray.remote(num_gpus=1)
# def squared(x):
#     time.sleep(1)
#     y = x**2
#     return y

# tic = time.perf_counter()

# lazy_values = [squared.remote(x) for x in range(10)]
# values = ray.get(lazy_values)

# toc = time.perf_counter()

# print(f'Elapsed time {toc - tic:.2f} s')
# print(f'{values[:5]} ... {values[-5:]}')

# ray.shutdown()