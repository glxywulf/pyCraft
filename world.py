import numpy as np
import math
import heapq
import render
from math import cos, sin
from numpy import ndarray
from typing import NamedTuple, List, Any, Tuple, Optional

# object that represents a chunk's position in terms of the world space
class ChunkPosition(NamedTuple):
    x: int
    y: int
    z: int

# even further specification on block coords in 3d world space
class BlockPosition(NamedTuple):
    x: int
    y: int
    z: int

# what kind of block a block is
blockID = str

# chunk object
class Chunk:
    pos : ChunkPosition
    blocks = ndarray
    lightlvls = ndarray
    instances : list[Any]
    
    isFinalized : bool = False
    isTicking : bool = False
    isVisible : bool = False
    
    # constructor, chunk should only have a positional instance variable made up of ChunkPositions
    def __init__(self, position : ChunkPosition):
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
                    self.setBlock(app, BlockPosition(x, y, z), bID, doUpdateLight = False, doUpdateBuried = False)
    
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
        (xw, yw, zw) = self.blocks.shape
        (x, y, z) = position
        return x * yw * zw + y * zw + z
    
    # takes in a block id and converts it back to its coordinate position
    def coordFromID(self, id):
        (x, y, z) = self.blocks.shape

        xID = id // (y + z)
        yID = (id // z) % y
        zID = id % z
        
        return BlockPosition(xID, yID, zID)
    
    # returns a block's position with respect to the entire world instead of 
    # just the current chunk
    def _globalBlockPos(self, blockPos):
        (x, y, z) = blockPos
        
        x += 16 * self.position[0]
        y += 16 * self.position[1]
        z += 16 * self.position[2]
        
        return BlockPosition(x, y, z)
    
    def updateBuried(self, app, bp):
        # get the blocks id which translates to its index in the instance list
        id = self.coordToID(bp)
        
        # if the blocks instance is None, return to kill the function
        if(self.instances[id] is None):
            return
        
        # get the specific block position with regards to the world.
        gloPos = self._globalBlockPos(bp)
        
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
    def setBlock(self, app, bp, bID, doUpdateLight = True, doUpdateBuried = True):
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
            [mX, mY, mZ] = blockInWorld(self._globalBlockPos(bp))
            
            # set the blocks instance in self.instances as a list that contains the Instance object from
            # render file, and a boolean which represents whether or not the block is buried
            self.instances[id] = [render.Instance(app.cube, np.array([[mX], [mY], [mZ]]), texture), False]
            
            # check if we need to update whether block is buried or not
            if(doUpdateBuried):
                self.updateBuried(app, bp)
        
        # get global block position
        gloPos = self._globalBlockPos(bp)
        
        # if we need to update light and buried states on a global scale then we
        # update the values
        if(doUpdateBuried):
            updateBuriedNear(app, gloPos)
                
        if(doUpdateLight):
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
    (chunk, _) = localChunk(bp)
    
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
        # this checks if its buried
        if not coordOccupied(app, adjaBlockPos(bp, fID)):
            return False
    
    # all spaces adjacent to block are taken so block is buried
    return True

# update block buried booleans for blocks that are next to another block if that
# coord is in a chunk that's stored in app.chunks
def updateBuriedNear(app, bp):
    for fID in range(0, 12, 2):
        adjaPos =  adjaBlockPos(bp, fID)
        if coordInBound(app, adjaPos):
            updateBuried(app, adjaPos)

# will return a generator object full of every chunk adjacent to inputted chunkPos
def adjaChunk(cp, dist):
    for xOffset in range(-dist, dist + 1):
        for zOffset in range(-dist, dist + 1):
            if(xOffset == zOffset):
                continue
            
            (x, y, z) = cp
            
            x += xOffset
            z += zOffset
            
            newChunkPos = ChunkPosition(x, y, z)
            
            yield newChunkPos

# unload specified chunk at position cp by removing it from app.chunks
# helper function to unload chunks
def unloadChunk(app, cp):
    print(f"Unloading chunk at {cp}")
    app.chunks.pop(cp)

# load chunk by adding new chunk into app.chunks
# helper function to load chunks
def loadChunk(app, cp):
    print(f"Loading chunk at {cp}")
    app.chunks[cp] = Chunk(cp)
    app.chunks[cp].generate(app)

# mass load and unload all chunks
def loadUnloadChunk(app):
    (cp, _) = localChunk(nearestBP(app.camPos[0], app.camPos[1], app.camPos[2]))
    (x, _, z) = cp
    
    shouldUnload = []
    
    # get and put chunks that need to be unloaded into shouldUnload list
    for unloadCP in app.chunks:
        (ux, _, uz) = unloadCP
        dist = max(abs(ux - x), abs(uz - z))

        # unload chunk
        if(dist > 2):
            shouldUnload.append(unloadCP)
    
    # once done getting chunks to unload, unload them all
    for unloadCP in shouldUnload:
        unloadChunk(app, unloadCP)
        
    loadedChunks = 0
    
    for loadCP in adjaChunk(cp, 2):
        if(loadCP not in app.chunks):
            (ux, _, uz) = unloadCP
            dist = max(abs(ux - x), abs(uz - z))
            
            urgent = (dist <= 1)
            
            if (urgent or (loadedChunks < 1)):
                loadedChunks += 1
                loadChunk(app, loadCP)

# return the number of loaded adjacent chunks
def countLoadedAdjaChunk(app, cp, dist):
    count = 0
    
    for pos in adjaChunk(cp, dist):
        if(pos in app.chunks):
            count += 1
            
    return count

# light and optimize chunk if chunk is finalized and its next to 8 chunks
def tickChunk(app):
    for cp in app.chunks:
        chunk = app.chunks[cp]
        adjaChunks = countLoadedAdjaChunk(app, cp, 1)
        
        if not (chunk.isFinalized and adjaChunks != 8):
            chunk.lightOpt(app)
            
        chunk.isVisible = (adjaChunks == 8)
        chunk.isTicking = (adjaChunks == 8)
        
# Ticking is done in stages so that collision detection works as expected:
# First we update the player's Y position and resolve Y collisions,
# then we update the player's X position and resolve X collisions,
# and finally update the player's Z position and resolve Z collisions.
def tick(app):
    loadUnloadChunk(app)
    
    tickChunk(app)
    
    app.camPos[1] += app.playerVelocity[1]
    
    if(app.playerOnGround):
        if not(hasBeneath(app)):
            app.playerOnGround = False
        else:
            app.playerVelocity[1] = -app.gravity
            [_, yP, _] = app.camPos
            
            yP -= app.playerHeight
            yP -= .1
            feetPos = round(yP)
            
            if(hasBeneath(app)):
                app.playerOnGround = True
                app.playerVelocity[1] = 0
                app.camPos[1] = (feetPos + .5) + app.playerHeight
    
    # W makes the player go forward, S makes them go backwards,
    # and pressing both makes them stop!
    z = float(app.w) - float(app.s)
    # Likewise for side to side movement
    x = float(app.d) - float(app.a)
    
    if(x != 0 or z != 0):
        mag = math.sqrt(x * x + z * z)
        x /= mag
        z /= mag
        
        newX = math.cos(app.camYaw) * x - math.sin(app.camYaw) * z
        newZ = math.sin(app.camYaw) * x + math.cos(app.camYaw) * z
        
        x, z = newX, newZ
        
        x *= app.playerWalkSpeed
        z *= app.playerWalkSpeed
        
    xVel = x
    zVel = z
    
    minY = round((app.camPos[1] - app.playerHeight + 0.1))
    maxY = round((app.camPos[1]))
    
    # check for x axis collisions and respond appropriately
    app.camPos[0] += xVel
    
    for y in range(minY, maxY):
        for z in [app.camPos[2] - app.playerRadius * 0.99, app.camPos[2] + app.playerRadius * 0.99]:
            x = app.camPos[0]
            
            hiXCoord = round((x + app.playerRadius))
            loXCoord = round((x - app.playerRadius))
            
            # if collision on right, move player back left
            if(coordOccupied(app, BlockPosition(hiXCoord, y, round(z)))):
                xEdge = (hiXCoord - .5)
                app.camPos[0] = xEdge - app.playerRadius
            
            # if collision on left, move player back right
            elif(coordOccupied(app, BlockPosition(loXCoord, y, round(z)))):
                xEdge = (loXCoord + .5)
                app.camPos[0] = xEdge + app.playerRadius
    
    # now we're gonna do the same thing, except for the z axis stuffs
    app.camPos[2] += zVel
    
    for y in range(minY, maxY):
        for z in [app.camPos[2] - app.playerRadius * 0.99, app.camPos[2] + app.playerRadius * 0.99]:
            Z = app.camPos[0]
            
            hiZCoord = round((Z + app.playerRadius))
            loZCoord = round((Z - app.playerRadius))
            
            # if collision on behind, move player back forward
            if(coordOccupied(app, BlockPosition(hiZCoord, y, round(z)))):
                ZEdge = (hiZCoord - .5)
                app.camPos[0] = ZEdge - app.playerRadius
            
            # if collision on front, move player back backword
            elif(coordOccupied(app, BlockPosition(loZCoord, y, round(z)))):
                ZEdge = (loZCoord + .5)
                app.camPos[0] = ZEdge + app.playerRadius

# access chunk object at specified position and set it's lightlvl to inputted lvl
def setLight(app, bp, lvl):
    (chunk, (x, y, z)) = getChunk(app, bp)
    chunk.lightlvls[x, y, z] = lvl

# update light levels for blocks
def updateLight(app, bp):
    # FIXME: Will be changed later since it bugs out a bit. Doesn't quite propogate
    # over chunk boundaries without making it much much slower

    # get the chunk that we're in and initialize the lightlvls list attribute of the chunk
    (chunk, bp) = getChunk(app, bp)
    chunk.lightlvls = np.full_like(chunk.blocks, 0, int)
    
    # get the shape of chunk.blocks
    shape = chunk.blocks.shape
    
    # keep a list of updated light stuff and blocks that are queueed up to be updated
    done = []
    queue = []
    
    # put the blocks into queue in heap order
    for x in range(shape[0]):
        for z in range(shape[2]):
            y = shape[1] - 1
            heapq.heappush(queue, (-7, BlockPosition(x, y, z)))
    
    # while the queue still has blocks waiting in it
    while (len(queue) > 0):
        (light, pos) = heapq.heappop(queue)
        light *= -1
        
        # if we've seen the position before then ignore it and continue with the next block
        if(pos in done):
            continue
        
        # if we haven't seen it, append it into done list
        done.append(pos)
        
        # get the block postion and set the lightlvl in the same position to the light lvl
        (x, y, z) = pos
        chunk.lightlvls[x, y, z] = light
        
        for fID in range(0, 12, 2):
            nP = adjaBlockPos(pos, fID)
            gP = chunk._globalBlockPos(nP)
            
            # if the next block position is out of bounds or occupied or out of chunk
            # just ignore it and keep its lightlvl the same
            if(nP in done):
                continue
            if not coordInBound(app, gP):
                continue
            if coordOccupied(app, gP):
                continue
            if(nP[0] < 0 or nP[0] >= 16):
                continue
            if(nP[1] < 0 or 16 <= nP[1]):
                continue
            if(nP[2] < 0 or 16 <= nP[2]):
                continue
            
            # if it passes all of those checks, check if light is 7 and the face
            # that we're checking that's 7 is the top face. set lights next to it to 7
            if(light == 7 and fID == 8):
                nLight = 7
            
            # otherwise, next light lvl should be the max of light - 1 and 0
            else:
                nLight = max(light - 1, 0)
            
            # push the next light and next position into the queue
            heapq.heappush(queue, (-nLight, nP))
        
# remove blocks from the world and replace with 'air' block
def removeBlock(app, bp):
    setBlock(app, bp, 'air')

# set block function that'll place blocks
def addBlock(app, bp, bID):
    setBlock(app, bp, bID)

# check if player has block beneath them or not. returns boolean
def hasBeneath(app):
    xP, yP, zP = app.camPos
    yP -= app.playerHeight
    yP -= .1
    
    for x in [xP - app.playerRadius * 0.99, xP + app.playerRadius * 0.99]:
        for z in [zP - app.playerRadius * 0.99, zP + app.playerRadius * 0.99]:
            feetPos = nearestBP(x, yP, z)
            if(coordOccupied(app, feetPos)):
                return True
            
    return False

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

# * needs to be optimized later too
# returns a tuple containing the block position of a block we're looking at
# and which face we're looking at
def lookBlock(app):
    lookX = cos(app.camPitch) * sin(-app.camYaw)
    lookY = sin(app.camPitch)
    lookZ = cos(app.camPitch) * cos(-app.camYaw)
    
    # magnification thing
    mag = math.sqrt(lookX ** 2 + lookY ** 2 + lookZ ** 2)
    lookX /= mag
    lookY /= mag
    lookZ /= mag
    
    step = .1
    lookX *= step
    lookY *= step
    lookZ *= step
    
    [x, y, z] = app.camPos
    
    maxDist = .6
    
    bp = None
    
    for _ in range(int(maxDist / step)):
        x += lookX
        y += lookY
        z += lookZ
        
        tempBP = nearestBP(x, y, z)
        
        if(coordOccupied(app, tempBP)):
            bp = tempBP
            break
        
    if(bp is None):
        return None
    
    [cx, cy, cz] = bp
    
    x -= cx
    y -= cy
    z -= cz
    
    if(abs(x) > abs(y) and abs(x) > abs(z)):
        face = 'right' if x > 0.0 else 'left'
    elif(abs(y) > abs(x) and abs(y) > abs(z)):
        face = 'top' if y > 0.0 else 'bottom'
    else:
        face = 'front' if z > 0.0 else 'back'

    return (bp, face)
    