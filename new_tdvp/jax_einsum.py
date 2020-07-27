#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import numpy as onp
import jax.numpy as np
from jax import jit, device_put

U1 = onp.random.rand(16).reshape(2,2,2,2)
U4 = onp.random.rand(2,2)
U3 = onp.random.rand(16**2).reshape(2,2,2,2,2,2,2,2)

device_put_times = {"U1":[], "U4":[], "U3":[]}

for i in range(1000):
    start1 = time.time()
    U1d = device_put(U1)
    end1 = time.time()
    
    start2 = time.time()
    U4d = device_put(U4)
    end2 = time.time()
    
    start3 = time.time()
    U3d = device_put(U3)
    end3 = time.time()
    
    device_put_times["U1"].append(-start1 + end1)
    device_put_times["U4"].append(-start2 + end2)
    device_put_times["U3"].append(-start3 + end3)


path_jax = np.einsum_path(
        U1, [6,7,26,27],
        U1, [8,9,28,29],
        U1, [10,11,30,31],
        U1, [27,28,22,23],
        U1, [29,30,24,25],
        U3,[22,23,24,25,18,19,20,21],
        U4, [26,12],
        U4, [31,17],
        U1, [18,19,13,14],
        U1, [20,21,15,16],
        U1, [12,13,0,1],
        U1, [14,15,2,3],
        U1, [16,17,4,5],
        [0,1,2,3,4,5,6,7,8,9,10,11],
        optimize = "greedy"
    )[0]

p1 = np.einsum_path(
        U1, [6,7,26,27],
        U1, [8,9,28,29],
        U1, [10,11,30,31],
        U1, [27,28,22,23],
        U1, [29,30,24,25],
        U3,[22,23,24,25,18,19,20,21],
        U4, [26,12],
        U4, [31,17],
        U1, [18,19,13,14],
        U1, [20,21,15,16],
        U1, [12,13,0,1],
        U1, [14,15,2,3],
        U1, [16,17,4,5],
        [0,1,2,3,4,5,6,7,8,9,10,11],
        optimize = "greedy"
    )


def overlap_jax(U1,U2,U3,U4,U5,U6):
    p = np.einsum(
        U1, [6,7,26,27],
        U1, [8,9,28,29],
        U1, [10,11,30,31],
        U2, [27,28,22,23],
        U2, [29,30,24,25],
        U3,[22,23,24,25,18,19,20,21],
        U4, [26,12],
        U4, [31,17],
        U5, [18,19,13,14],
        U5, [20,21,15,16],
        U6, [12,13,0,1],
        U6, [14,15,2,3],
        U6, [16,17,4,5],
        [0,1,2,3,4,5,6,7,8,9,10,11],
        optimize = path_jax
        )[0,0,0,0,0,0,0,0,0,0,0,0]
    
    return p


def overlap_num(U1,U2,U3,U4,U5,U6):
    p = onp.einsum(
        U1, [6,7,26,27],
        U1, [8,9,28,29],
        U1, [10,11,30,31],
        U2, [27,28,22,23],
        U2, [29,30,24,25],
        U3,[22,23,24,25,18,19,20,21],
        U4, [26,12],
        U4, [31,17],
        U5, [18,19,13,14],
        U5, [20,21,15,16],
        U6, [12,13,0,1],
        U6, [14,15,2,3],
        U6, [16,17,4,5],
        [0,1,2,3,4,5,6,7,8,9,10,11],
        optimize = "greedy"
        )[0,0,0,0,0,0,0,0,0,0,0,0]
    
    return p

jit_overlap = jit(overlap_jax)

jit_times = {"With Device":[], "Without Device": [], "Numpy Raw":[]}

for i in range(1000):
    start1 = time.time()
    a = overlap_num(U1,U1,U3,U4,U1,U1)
    end1 = time.time()
    
    start2 = time.time()
    a = jit_overlap(U1,U1,U3,U4,U1,U1).block_until_ready()
    end2 = time.time()

    start3 = time.time()
    a = jit_overlap(U1d,U1d,U3d,U4d,U1d,U1d).block_until_ready()
    end3 = time.time()
    
    jit_times["With Device"].append(end3 - start3)
    jit_times["Without Device"].append(end2 - start2)
    jit_times["Numpy Raw"].append(end1 - start1)

mean_raw = onp.mean(jit_times["Numpy Raw"])
mean_no_device = onp.mean(jit_times["Without Device"])
mean_device = onp.mean(jit_times["With Device"])
d1m = onp.mean(device_put_times["U1"])
d4m = onp.mean(device_put_times["U4"])
d3m = onp.mean(device_put_times["U3"])

def to_sf(num):
    "Print number in scientific format"
    return "{:.3e}".format(num)

os = ""

os += f"The numpy function takes time: {to_sf(mean_raw)}\n"
os += f"The jitted jax func takes time: {to_sf(mean_no_device)}\n"
os += f"Put Tensors on device:\n{to_sf(d1m)}\n{to_sf(d4m)}\n{to_sf(d3m)}\nTotal: {to_sf(d1m+d4m+d3m)}\n"
os += f"With device tensors: {to_sf(mean_device)}\n"
os += f"Total time on device: {to_sf(mean_device + d1m + d4m + d3m)}"

print(os)


