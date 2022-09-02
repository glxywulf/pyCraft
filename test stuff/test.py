from typing import NamedTuple, List, Any, Tuple, Optional
from numpy import ndarray
import numpy as np

class Thing(NamedTuple):
    
    x: str # format for class instance variables 
    y: str
    
thing = Thing('he', 4)

# print(type(thing.y))

test = np.array([[1,2,3], [1,2,3]])

print(test.shape)

blocks: ndarray

blocks = np.full((3,4,3), 'air') 
# (x, y, z) = layers, rows, columns
# (x, y) = rows, columns

print(blocks)
print(blocks.shape)
print(blocks.size)