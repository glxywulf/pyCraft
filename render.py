import math
import time
import world
import numpy as np
from math import sin, cos
from numpy import ndarray
from typing import List, Tuple, Optional, Any
# from world import BlockPos, adjaBlockPos

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
    
    return np.dot()

def isFaceVisible(app, bp, fID):
    pass

def getBlockFaceLight(app, bp, fID):
    pass

def isBlockBack(app, bp, fID):
    pass

def clip(app, vertices, face):
    pass

def cullInstance(app, camMatrix, instance, bp):
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
