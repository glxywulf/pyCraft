from typing import NamedTuple, List, Any, Tuple, Optional
from numpy import ndarray
import numpy as np

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

test = np.array([[0], [0], [0]])
print(test)

def isPrime(n):
    return np.array([[0], [0], [0]])

def nthPrime(n):
    return np.array([[0], [0], [0]])

isPrime(9) @ nthPrime(22)

# ? weird syntax thing

# thing = Tuple[int, int, int]
# thing = (1,'hi',3)

# print(repr(thing[0]))
# print(f'{thing}')

# Color = str

# print(Color)
# print(thing)