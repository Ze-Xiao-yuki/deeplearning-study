#!/usr/bin/env python
# coding: utf-8

# # 异步计算
# :label:`sec_async`
# 
# 今天的计算机是高度并行的系统，由多个CPU核、多个GPU、多个处理单元组成。通常每个CPU核有多个线程，每个设备通常有多个GPU，每个GPU有多个处理单元。总之，我们可以同时处理许多不同的事情，并且通常是在不同的设备上。不幸的是，Python并不善于编写并行和异步代码，至少在没有额外帮助的情况下不是好选择。归根结底，Python是单线程的，将来也是不太可能改变的。因此在诸多的深度学习框架中，MXNet和TensorFlow之类则采用了一种*异步编程*（asynchronous programming）模型来提高性能，而PyTorch则使用了Python自己的调度器来实现不同的性能权衡。对PyTorch来说GPU操作在默认情况下是异步的。当调用一个使用GPU的函数时，操作会排队到特定的设备上，但不一定要等到以后才执行。这允许我们并行执行更多的计算，包括在CPU或其他GPU上的操作。
# 
# 因此，了解异步编程是如何工作的，通过主动地减少计算需求和相互依赖，有助于我们开发更高效的程序。这能够减少内存开销并提高处理器利用率。


import os
import subprocess
import numpy
import torch
from torch import nn
from d2l import torch as d2l


# GPU计算热身
device = d2l.try_gpu()
a = torch.randn(size=(1000, 1000), device=device)
b = torch.mm(a, a)

with d2l.Benchmark('numpy'):
    for _ in range(10):
        a = numpy.random.normal(size=(1000, 1000))
        b = numpy.dot(a, a)

with d2l.Benchmark('torch'):
    for _ in range(10):
        a = torch.randn(size=(1000, 1000), device=device)
        b = torch.mm(a, a)


with d2l.Benchmark():
    for _ in range(10):
        a = torch.randn(size=(1000, 1000), device=device)
        b = torch.mm(a, a)
    torch.cuda.synchronize(device)




x = torch.ones((1, 2), device=device)
y = torch.ones((1, 2), device=device)
z = x * y + 2



