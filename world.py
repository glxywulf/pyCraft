import numpy as np
import math
import heapq
import render
from math import cos, sin
from numpy import ndarray
from typing import NamedTuple, List, Any, Tuple, Optional

# object that represents a chunk's position in terms of the world space
class ChunkPosition(NamedTuple):
    x, y, z = int, int, int

# even further specification on block coords in 3d world space
class BlockPosition(NamedTuple):
    x, y, z = int, int, int

# what kind of block a block is
blockID = str

# chunk object
class Chunk:
    blocks = ndarray
    lightlvls = ndarray
    instances : list[Any]
    
    isFinalized : bool = False
    isTicking : bool = False
    isVisible : bool = False
    
    # constructor, chunk should only have a positional instance variable made up of ChunkPositions
    def __init__(self, position):
        self.position = position
    
    # this is going to be the bulk method that generates each block and gives them
    # their id and traits to be interpreted in render
    def generate(self, app):
        self.blocks = np.full((16, 16, 16), 'air')
        self.lightlvls = np.full((16, 16, 16), 7)
        self.instances = [None] * self.blocks.size
        
        for x in range(0, 16):
            for z in range(0, 16):
                for y in range(0, 8):
                    self.lightlvls[x, y, z] = 0
                    bID = 'grass' if (y == 7) else 'stone'
    
    # convert a block coordinate into a UID that specifically denotes a specific block
    def coordToID(self, position):
        (x, y, z) = position
        (xB, yB, zB) = self.blocks.shape
        
        return x * yB * zB + y * zB + z
    
    # takes in a block id and converts it back to its coordinate position
    def coordFromID(self, id):
        (x, y, z) = self.blocks.shape

        xID = id // (y + z)
        yID = (id // z) % y
        zID = id % z
        
        return BlockPosition(xID, yID, zID)
    
    # returns a block's position with respect to the entire world instead of 
    # just the current chunk
    def globalBlockPos(self, blockPos):
        (x, y, z) = blockPos
        
        x += 16 * self.position[0]
        y += 16 * self.position[1]
        z += 16 * self.position[2]
        
        return BlockPosition(x, y, z)
    
    def setBlock(self):
        pass