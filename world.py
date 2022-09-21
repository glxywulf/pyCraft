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
    
    # set the visible faces of every single block in every chunk and finalize block face states
    def lightOpt(self, app):
        print(f"Lighting and optimizing chunk at {self.position}")
        
        for xID in range(0, 16):
            for yID in range(0, 8):
                for zID in range(0, 16):
                    self.updateBuried(app, BlockPosition(xID, yID, zID))
        
        self.isFinalized = True
    
    # make an interable generator object that contains all of the Blocks(specified
    # by position) and their instances.
    def iterateInstances(self):
        if(self.isFinalized and self.isVisible):
            for (i, instance) in enumerate(self.instances):
                if(instance != None):
                    wx = self.position[0] * 16 + (i // 256)
                    wy = self.position[1] * 16 + (i // 16) % 16
                    wz = self.position[2] * 16 + (i % 16)
                    yield (BlockPosition(wx, wy, wz), instance)
    
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
    
    def updateBuried(self, app, bp):
        # get the blocks id which translates to its index in the instance list
        id = self.coordToID(bp)
        
        # if the blocks instance is None, return to kill the function
        if(self.instances[id] == None):
            return
        
        # get the specific block position with regards to the world.
        gloPos = self.globalBlockPos(bp)
        
        buried = False
        for fID in range(0, 12, 2):
            # get the coords of the adjacent block in the direction of the face
            adjPos = adjaBlockPos(gloPos, fID)
            
            # if a block occupies the coord that's adjacent to the block we're checking
            # set the visibleFace attribute of the specific instance to false
            # which basically sets a blocks face to be invisible
            if coordOccupied(app, adjPos):
                self.instances[id][0].visibleFaces[fID] = False
                self.instances[id][0].visibleFaces[fID + 1] = False
                pass
            # if there isn't a block adjacent to a block's face then the block's
            # face should be visible
            else:
                self.instances[id][0].visibleFaces[fID] = True
                self.instances[id][0].visibleFaces[fID + 1] = True
                buried = True
        
        self.instances[id][1] = buried
    
    # check if a certain coordinate is occupied by a block
    def coordOccupied(self, bp):
        # get the block's coordinate position
        (x, y, z) = bp
        
        # check if it's an 'air' block. if not return True since that spot is occupied
        return self.blocks[x, y, z] != 'air'
    
    # initializes the block fully, assigning the block its position, type, instances,
    # light lvls, buried states, i think that's it.
    def setBlock(self, app, bp, bID, doLightUpdate = True, doBuriedUpdate = True):
        # set the block to bID since that's what type it is
        (x, y, z) = bp
        self.blocks[x, y, z] = bID
        
        # instance index based off of block position
        id = self.coordToID(bp)
        
        # if the block is going to be an 'air' block then it doesn't need instances
        if(bID == 'air'):
            self.instances[id] = None
        # otherwise, we set up the instances here
        else:
            # set its texture by using app's texture dictionary
            texture = app.textures[bID]
            
            # get the models x/y/z stuffs
            [mX, mY, mZ] = blockInWorld(self.globalBlockPos(bp))
            
            # set the blocks instance in self.instances as a list that contains the Instance object from
            # render file, and a boolean which represents whether or not the block is buried
            self.instances[id] = [render.Instance(app.cube, np.array([[mX], [mY], [mZ]]), texture), False]
            
            # check if we need to update whether block is buried or not
            if(doBuriedUpdate):
                self.updateBuried(app, bp)
        
        # get global block position
        gloPos = self.globalBlockPos(bp)
        
        # if we need to update light and buried states on a global scale then we
        # update the values
        if(doBuriedUpdate):
            updateBuriedNear(app, gloPos)
                
        if(doLightUpdate):
            updateLight(app, gloPos)

# * Helper Functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# accesses the class function inside chunk.
def updateBuried(app, bp):
    (chunk, innerPos) = getChunk(app, bp)
    chunk.updateBuried(app, innerPos)

# get the chunk a certain block is in
def getChunk(app, bp):
    # chunk coords are div 16 of the block coords, make it into chunkPos object
    (chunkX, chunkY, chunkZ) = bp
    
    chunkX //= 16
    chunkY //= 16
    chunkZ //= 16
    
    chunk = app.chunks[ChunkPosition(chunkX, chunkY, chunkZ)]
    
    # get the block position with respect to the chunk
    (x, y, z) = bp
    x %= 16
    y %= 16
    z %= 16
    
    # return the chunk and blockPos in the chunk
    return (chunk, BlockPosition(x, y, z))

# helper to get the variables we need for the function inside the Chunk class
def coordOccupied(app, bp):
    # if the blockPos is outside of chunk bounds return False
    if not coordInBound(app, bp):
        return False
    
    (chunk, innerBP) = getChunk(app, bp)
    
    return chunk.coordOccupied(innerBP)

# access the setBlock class function of chunk outside of chunk object
def setBlock(app, bp, bID, doUpdateLight = True):
    (chunk, innerBP) = getChunk(app, bp)
    chunk.setBlock(app, innerBP, bID, doUpdateLight)
    
# returns the chunk's world position based off of a global block position
def localChunk(bp):
    (x, y, z) = bp
    
    # chunk coords should be what ever the xyz is int divided by 16
    chunkX = x // 16
    chunkY = y // 16
    chunkZ = z // 16
    
    # create the ChunkPosition object
    cPos = ChunkPosition(chunkX, chunkY, chunkZ)
    
    # now we need the block's position with respect to the chunk that it's in
    x %= 16
    y %= 16
    z %= 16
    
    newBP = BlockPosition(x, y, z)
    
    return (cPos, newBP)

# figure out if the coords of the block are within the chunk
def coordInBound(app, bp):
    (chunk, block) = localChunk(bp)
    
    return chunk in app.chunks

# returns integer value of the closest block in a certain axis 
# (i.e. x: 14.2345 -> 14; y: 1.235 -> 1)
def nearestBlock(coord):
    return round(coord)

# reuturns the coordinate of the nearest block
def nearestBP(x, y, z):
    bX = nearestBlock(x)
    bY = nearestBlock(y)
    bZ = nearestBlock(z)
    return BlockPosition(bX, bY, bZ)

# returns the position of the center of a block in relation to the world
def blockInWorld(bp):
    (x, y, z) = bp
    
    return (x, y, z)

# returns a boolean value that represents whether or not a block is fully buried or not
def isBuried(app, bp):
    # if any of the faces of the block isn't adjacent to anything, then block isn't buried
    for fID in range(0, 12, 2):
        adja = adjaBlockPos(bp, fID)
        
        # this checks if its buried
        if not coordOccupied(app, adja):
            return False
    
    # all spaces adjacent to block are taken so block is buried
    return True

# TODO Code this

def updateBuriedNear(app, bp): # ? Priority
    pass

def adjaChunk(cp, dist):
    pass

def unloadChunk(app, cp):
    pass

def loadChunk(app, cp):
    pass

def loadUnloadChunk(app):
    pass

def countLoadChunk(app, cp, dist):
    pass

def tickChunk(app):
    pass

def tick(app):
    pass

def setLight(app, bp, lvl):
    pass

def updateLight(app, bp): # ? Priority
    pass

def removeBlock(app,  bp):
    pass

def addBlock(app, bp):
    pass

def hasBeneath(app):
    pass

# TODO Code this

# returning the position of the block that is next to the original block
# that is touching the face that was inputted
def adjaBlockPos(bP, fID):
    # get the blocks position in a list
    x, y, z = bP
    
    # turns whatever fID is into half of what it was so that it's a value within
    # the index of the list below
    fID //= 2
    
    # Left, right, front, back, bottom, top
    (a, b, c) = [(-1, 0, 0), (1, 0, 0), (0, 0, -1), (0, 0, 1), (0, -1, 0), (0, 1, 0)][fID]
    
    # the original coords + the face tuples translate the coords to be the coords
    # of the block that is adjacent in the direction of the inputted face
    x += a
    y += b
    z += c
    
    return BlockPosition(x, y, z)

# TODO Code this

def lookBlock(app):
    pass

# TODO Code this
