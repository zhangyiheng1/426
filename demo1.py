"""
author:张怡恒
"""
import torch

print(torch.version.cuda)	#查看cuda版本
print(torch.cuda.is_available())  # 查看cuda是否可用
print(torch.cuda.device_count())  # 查看可行的cuda数目

