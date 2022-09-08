from typing import NamedTuple, List, Any, Tuple, Optional
from numpy import ndarray
import numpy as np
import matplotlib.pyplot as plt
from perlin_noise import *

# class Thing(NamedTuple):
    
#     x: str # format for class instance variables 
#     y: str
    
# thing = Thing('he', 4)

# print(type(thing.y))

# ? ndarray practice stuff

# test = np.array([[1,2,3], [1,2,3]])

# print(test.shape)

# blocks: ndarray

# blocks = np.full((3,4,3), 'air') 
# # (x, y, z) = layers, rows, columns
# # (x, y) = rows, columns

# print(blocks)
# print(blocks.shape)
# print(blocks.size)

# test = [1,2,3,4,5,6]
# print(test[ :-1] / test[-1])

# cartesian : ndarray
# cartesian = np.array([[cartesian[0, 0]], [cartesian[1, 0]], [cartesian[2, 0]], [1.0]])

# assert(cartesian.shape[1] == 1)

# test = np.array([[0], [0], [0]])
# print(test)

# def isPrime(n):
#     return np.array([[0], [0], [0]])

# def nthPrime(n):
#     return np.array([[0], [0], [0]])

# isPrime(9) @ nthPrime(22)

# ? weird syntax thing

# thing = Tuple[int, int, int]
# thing = (1,'hi',3)

# print(repr(thing[0]))
# print(f'{thing}')

# Color = str

# print(Color)
# print(thing)

# test = (1,2,3)

# print(len(test))

# thing = [None] * 20

# print(list(enumerate(thing)))

# for (i, instance) in (enumerate(thing)):
#     print((i, instance))

# noise1 = PerlinNoise(octaves=25)
# noise2 = PerlinNoise(octaves=50)
# noise3 = PerlinNoise(octaves=75)
# noise4 = PerlinNoise(octaves=100)

# width = 500
# height = 500

# pic = []

# for i in range(width):
#     row = []
#     for j in range(height):
#         noise_val = noise1([i/width, j/height])
#         noise_val += 0.5 * noise2([i/width, j/height])
#         noise_val += 0.25 * noise3([i/width, j/height])
#         noise_val += 0.125 * noise4([i/width, j/height])
        
#         row.append(noise_val)
    
#     pic.append(row)
    
# plt.imshow(pic, cmap = 'gray')
# plt.show()