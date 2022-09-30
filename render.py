import math
import time
import world
import numpy as np
from math import sin, cos
from numpy import ndarray
from typing import List, Tuple, Optional, Any

class Model:
    def __init__(self, vertices, faces):
        self.vertices = vertices
        self.faces = []
        
        for face in faces:
            if(len(face) == 3):
                self.faces.append(face)
            else:
                raise Exception("Invalid face")
            
class Instance:
    def __init__(self, model, translation, texture):
        self.model = model
        self.trans = translation
        self.texture = texture
        self.vertices = list(map(turnToMatRow, self.transVert()))
        self.visibleFaces = [True] * len(model.faces)
    
    # returns list of instance vertices
    def getVertices(self):
        return self.vertices
    
    # returns list of translated vertices    
    def transVert(self):
        # result variable list
        result = []
        
        # for every vertice add the translation value and then append into result list
        for vertice in self.model.vertices:
            result.append(vertice + self.trans)
        
        return result
    
# turns an inputted array into a ndarray that represents a row in a matrix.
# * toHomogenous
def turnToMatRow(a):
    assert(a.shape[1] == 1)
    
    return np.array([[a[0, 0]], [a[1, 0]], [a[2, 0]], [1]])

# turns a matrix row into a cartesian coordinate
# * toCartesian
def matRowToCoord(a):
    # make sure the inputted array has one column
    assert(a.shape[1] == 1)
    
    # flatten the array into a 1d list of the values
    a = a.ravel()
    
    # return the first 3 values divided by the last value to get coordinates
    # that are a cartesian representation of the vertice
    return a[ :-1] / a[-1]

# returns a array with values of a vector that's been translated about the x-axis
# in the form of a matrix
# formula from: https://www.redcrab-software.com/en/Calculator/4x4/Matrix/Rotation-X
def rotateX(theta):
    return np.array([
        [1, 0, 0, 0],
        [0, cos(theta), -sin(theta),0],
        [0, sin(theta), cos(theta), 0],
        [0, 0, 0, 1]
    ])

# returns a array with values of a vector that's been translated about the y-axis
# in the form of a matrix
# formula from: https://www.redcrab-software.com/en/Calculator/4x4/Matrix/Rotation-Y
def rotateY(theta):
    return np.array([
        [cos(theta), 0, sin(theta), 0],
        [0, 1, 0, 0],
        [-sin(theta), 0, cos(theta), 0],
        [0, 0, 0, 1]
    ])

# returns a array with values of a vector that's been translated about the z-axis
# in the form of a matrix
# formula from: https://www.redcrab-software.com/en/Calculator/4x4/Matrix/Rotation-Z
def rotateZ(theta):
    return np.array([
        [cos(theta), -sin(theta), 0, 0],
        [sin(theta), cos(theta), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

# returns a matrix that holds translation values for the vertice
def transMatrix(x, y, z):
    return np.array([
        [1, 0, 0, x],
        [0, 1, 0, y],
        [0, 0, 1, z],
        [0, 0, 0, 1]
    ])
    
# 3d space to canvas matrix
def spaceCanvMat(camPos, yaw, pitch, vpDist, vpWidth, vpHeight, canvWidth, canvHeight):
    vp = vpCanvMatrix(vpWidth, vpHeight, canvWidth, canvHeight)
    
    return vp @ camToVpMatrix(vpDist) @ spaceToCameraMatrix(camPos, yaw, pitch)

# converts the matrix of the point in space into a matrix that the camera understands
# Original technique from
# https://gamedev.stackexchange.com/questions/168542/camera-view-matrix-from-position-yaw-pitch-worldup
# Modified similarly to: https://github.com/SuperTails/112craft 
def spaceToCameraMatrix(cp, y, p):
    y = -y
    a = cp[0]
    b = cp[1]
    c = cp[2]
    
    return np.array([
        [cos(y), 0, -sin(y), (c * sin(y)) - (a * cos(y))],
        [-sin(p) * sin(y), cos(p), -sin(p) * cos(y), (c * sin(p) * cos(y)) + (a * sin(p) * sin(y)) - (b * cos(p))],
        [cos(p) * sin(y), sin(p), cos(p) * cos(y), (-b * sin(p)) - (a * sin(y) * cos(p)) - (c * cos(y) * cos(p))],
        [0, 0, 0, 1]
    ])

# viewpoint with respect to canvas matrix
def vpCanvMatrix(vpW, vpH, cW, cH):
    w = cW / vpW
    h = cH / vpH
    
    x = cW * .5
    y = cH * .5
    
    return np.array([
        [w, 0, x],
        [0, h, y],
        [0, 0, 1]
    ])

# return a tuple of the distance from a point to the camera position
def spaceToCamera(point, cp):
    x = point[0] - cp[0]
    y = point[1] - cp[1]
    z = point[2] - cp[2]

    return [x, y, z]

# returns the distance of a camera to viewpoint in a matrix
def camToVpMatrix(vpd):
    return np.array([
        [vpd, 0, 0, 0],
        [0, vpd, 0, 0],
        [0, 0, 1, 0]
    ])
    
# camera to viewpoint
def camToViewpoint(point, vpd):
    # view point x and y calculated and divided based on z coord
    vpX = point[0] * vpd / point[2]
    vpY = point[1] * vpd / point[2]

    return [vpX, vpY]

# TODO Code stuffs ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# translate the player viewpoint from the xy point to a xy point on the canvas
def viewPointToCanv(point, vpW, vpH, canvWidth, canvHeight):
    canvX = (point[0] / vpW + 0.5) * canvWidth
    canvY = (-point[1] / vpH + 0.5) * canvHeight

    return [canvX, canvY]

# convert a point in space into a xy coordinate in the canvas
def spaceToCanvas(app, point):
    row = turnToMatRow(point)
    matrix = spaceCanvMat(app.camPos, app.camYaw, app.camPitch, app.vpDist, 
                          app.vpWidth, app.vpHeight, app.width, app.height)
    result = matRowToCoord(matrix @ row)
    
    return result

# find the normal face with respect to the polygon being shown
def faceNormal(v0, v1, v2):
    coord0 = matRowToCoord(v0)
    coord1 = matRowToCoord(v1)
    coord2 = matRowToCoord(v2)
    
    a = coord1 - coord0
    b = coord2 - coord0
    
    result = np.cross(a, b)
    
    return result

# vertices have to be within the camera space
# technique from: https://en.wikipedia.org/wiki/Back-face_culling
def isBackFace(v0, v1, v2):
    normal = faceNormal(v0, v1, v2)
    coord0 = matRowToCoord(v0)
    
    return -np.dot(coord0, normal) >= 0

# check if a certain face of a block is visible. returns a bool
def isFaceVisible(app, bp, fID):
    (x, y, z) = world.adjaBlockPos(bp, fID)
    
    # if adjacent coords are occupied then face is not visible
    if(world.coordOccupied(app, world.BlockPosition(x, y, z))):
        return False
    
    return True
    
# get the light lvl of a blocks face
def getBlockFaceLight(app, bp, fID):
    # get the light lvl of a block adjacent to the block in the direction of the face
    position = world.adjaBlockPos(bp, fID)
    (chunk, (x, y, z)) = world.getChunk(app, position)
    
    return chunk.lightlvls[x, y, z]

# returns bool saying if a back face is a back face
def isBlockBack(app, bp, fID):
    # get the face from face id, block global position
    fID //= 2
    (x, y, z) = world.blockInWorld(bp)
    
    # distance from camera
    xDiff = app.camPos[0] - x
    yDiff = app.camPos[1] - y
    zDiff = app.camPos[2] - z
    
    # left
    if(fID == 0):
        return xDiff > -.5
    # right
    elif(fID == 1):
        return xDiff < .5
    # front
    elif(fID == 2):
        return zDiff > -.5
    # back
    elif(fID == 3):
        return zDiff < .5
    # bottom
    elif(fID == 4):
        return yDiff > -.5
    # top
    elif(fID == 5):
        return yDiff < .5

# FIXME: This doesn't conserve the CCW vertex ordering. See how he changes this later

def clip(app, vertices, face):
    # lambda function that checks whether or not face vertice is out of view
    outOfView = lambda id : vertices[id][2] < app.vpDist
    
    # get a numeric value holding how many faces are visible
    numFaceVisible = (not outOfView(face[0])) + ((not outOfView(face[1])) + (not outOfView(face[2])))
    
    # if there are 0 visible faces then return empty list
    if(numFaceVisible == 0):
        return []
    # if there are 3 visible faces
    elif(numFaceVisible == 3):
        return [face]
    
    # get the xyz coordinates of each vertice of the that is in view 
    [v0, v1, v2] = sorted(face, key = outOfView)
    
    [[x0], [y0], [z0], _] = vertices[v0]
    [[x1], [y1], [z1], _] = vertices[v1]
    [[x2], [y2], [z2], _] = vertices[v2]
    
    if(numFaceVisible == 2):
        xd = (x2 - x0) * (app.vpDist - z0) / (z2 - z0) + x0
        yd = (y2 - y0) * (app.vpDist - z0) / (z2 - z0) + y0

        xc = (x2 - x1) * (app.vpDist - z1) / (z2 - z1) + x1
        yc = (y2 - y1) * (app.vpDist - z1) / (z2 - z1) + y1
        
        dID = len(vertices)
        vertices.append(np.array([[xd], [yd], [app.vpDist], [1]]))
        cID = len(vertices)
        vertices.append(np.array([[xc], [yc], [app.vpDist], [1]]))

        face0 = (v0, v1, dID)
        face1 = (v0, v1, cID)
        
        return [face0, face1]
    else:
        xa = (x1 - x0) * (app.vpDist - z0) / (z1 - z0) + x0
        ya = (y1 - y0) * (app.vpDist - z0) / (z1 - z0) + y0

        xb = (x2 - x0) * (app.vpDist - z0) / (z2 - z0) + x0
        yb = (y2 - y0) * (app.vpDist - z0) / (z2 - z0) + y0

        aIdx = len(vertices)
        vertices.append(np.array([[xa], [ya], [app.vpDist], [1.0]]))
        bIdx = len(vertices)
        vertices.append(np.array([[xb], [yb], [app.vpDist], [1.0]]))

        clippedFace = (v0, aIdx, bIdx)

        return [clippedFace]

# This converts the instance's vertices to points in camera space, and then:
# For all blocks, the following happens:
#       - Faces pointing away from the camera are removed
#       - Faces that are hidden 'underground' are removed
#       - The color of each face is adjusted based on lighting
#       - ~~A "fog" is applied~~ #! NOT IMPLEMENTED!
# For anything else:
#       - Normal back face culling is applied
# 
# Then, the faces are clipped, which may remove, modify, or split faces
# Then a list of faces, their vertices, and their colors are returned
def cullInstance(app, camMatrix, instance, bp):
    vertices = list(map(lambda v : camMatrix @ v, instance.getVertices()))
    
    faces = []
    
    skipNext = False
    
    # go through every block and assign each block's face with its texture stored in its instance
    for (fID, (face, color)) in enumerate(zip(instance.model.faces, instance.texture)):
        # if skipNext is true then we skip the block we're on and set skip to False
        if(skipNext):
            skipNext = False
            continue
        
        if(bp is not None):
            if not (instance.visibleFaces[fID]):
                pass
            
            if(isBlockBack(app, bp, fID)):
                pass
            
            if not (isFaceVisible(app, bp, fID)):
                pass
            #! Working here
        else:
            pass
        
        

def isBlockVisible(app, bp):
    pass

def renderInstance(app, canvas):
    pass

def drawToFace(app):
    pass

def drawToCanvas(app, canvas, faces):
    pass

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# get a bunch of times 
frameTimes = [0.0] * 10
frameIndex = 0

def redrawAll(app, canvas):
    # get an initial time
    startTime = time.time()
    
    # creates the sky
    canvas.create_rectangle(0.0, 0.0, app.width, app.height, fill = '#0080FF')
    
    # cursor
    canvas.create_oval(app.width / 2 - 1, app.height / 2 - 1, app.width / 2 + 1, app.height / 2 + 1)
    
    # access the variables global in this file
    global frameTimes
    global frameIndex

    # final time
    endTime = time.time()
    
    # put a time into the time list and increment the index
    frameTimes[frameIndex] = (endTime - startTime)
    frameIndex += 1
    
    # resets the frame index when it reaches 10
    frameIndex %= len(frameTimes)
    
    # average the times in order to get the frame rate
    frameTime = sum(frameTimes) / len(frameTimes) * 1000.0

    canvas.create_text(11, 11, text=f'Frame Time: {frameTime:.2f}ms', anchor='nw')
    canvas.create_text(10, 10, text=f'Frame Time: {frameTime:.2f}ms', anchor='nw', fill='white')
    
# TODO Code stuffs ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
