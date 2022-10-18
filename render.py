import math
import time
import world
import numpy as np
from math import sin, cos
from numpy import ndarray
from typing import List, Tuple, Optional, Any
from world import BlockPos, adjacentBlockPos

Color = str

Face = Tuple[int, int, int]

class Model:
    vertices : List[ndarray]
    faces : List[Face]
    
    def __init__(self, vertices : List[ndarray], faces : List[Face]):
        self.vertices = vertices
        self.faces = []
        
        for face in faces:
            if(len(face) == 4):
                1 / 0
            elif(len(face) == 3):
                self.faces.append(face)
            else:
                raise Exception("Invalid number of vertices for face")
            
class Instance:
    model : Model
    trans : ndarray
    texture : List[Color]
    visibleFaces : List[bool]
    
    _worldSpaceVertices : List[ndarray]
    
    def __init__(self, model : Model, trans : ndarray, texture : List[Color]):
        self.model = model
        self.trans = trans
        self.texture = texture

        self._worldSpaceVertices = list(map(toHomogenous, self.worldSpaceVerticesUncached()))
        self.visibleFaces = [True] * len(model.faces)
    
    # returns list of instance vertices
    def worldSpaceVertices(self) -> List[ndarray]:
        return self._worldSpaceVertices
    
    # returns list of translated vertices    
    def worldSpaceVerticesUncached(self) -> List[ndarray]:
        # result variable list
        result = []
        
        # for every vertice add the translation value and then append into result list
        for vertice in self.model.vertices:
            result.append(vertice + self.trans)
        
        return result
    
# turns an inputted array into a ndarray that represents a row in a matrix.
def toHomogenous(cartesian : ndarray) -> ndarray:
    # assert(cartesian.shape[1] == 1)
    
    return np.array([[cartesian[0, 0]], [cartesian[1, 0]], [cartesian[2, 0]], [1.0]])

# turns a matrix row into a cartesian coordinate
def toCartesian(cartesian : ndarray) -> ndarray:
    # make sure the inputted array has one column
    assert(cartesian.shape[1] == 1)

    cart = cartesian.ravel()

    return cart[:-1] / cart[-1]

# returns a array with values of a vector that's been translated about the x-axis
# in the form of a matrix
# formula from: https://www.redcrab-software.com/en/Calculator/4x4/Matrix/Rotation-X
def rotateX(theta):
    return np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, math.cos(theta), -math.sin(theta), 0.0],
        [0.0, math.sin(theta), math.cos(theta), 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

# returns a array with values of a vector that's been translated about the y-axis
# in the form of a matrix
# formula from: https://www.redcrab-software.com/en/Calculator/4x4/Matrix/Rotation-Y
def rotateY(theta):
    return np.array([
        [math.cos(theta), 0.0, math.sin(theta), 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-math.sin(theta), 0.0, math.cos(theta), 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

# returns a array with values of a vector that's been translated about the z-axis
# in the form of a matrix
# formula from: https://www.redcrab-software.com/en/Calculator/4x4/Matrix/Rotation-Z
def rotateZ(theta):
    return np.array([
        [math.cos(theta), -math.sin(theta), 0.0, 0.0],
        [math.sin(theta), math.cos(theta), 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

# returns a matrix that holds translation values for the vertice
def translationMat(x, y, z):
    return np.array([
        [1.0, 0.0, 0.0, x],
        [0.0, 1.0, 0.0, y],
        [0.0, 0.0, 1.0, z],
        [0.0, 0.0, 0.0, 1.0]
    ])
    
# 3d space to canvas matrix
def wsToCanvasMat(camPos, yaw, pitch, vpDist, vpWidth, vpHeight, canvWidth, canvHeight):
    vpToCanv = vpToCanvasMat(vpWidth, vpHeight, canvWidth, canvHeight)
    return vpToCanv @ camToVpMat(vpDist) @ wsToCamMat(camPos, yaw, pitch)

def csToCanvasMat(vpDist, vpWidth, vpHeight, canvWidth, canvHeight):
    vpToCanv = vpToCanvasMat(vpWidth, vpHeight, canvWidth, canvHeight)
    return vpToCanv @ camToVpMat(vpDist)

# converts the matrix of the point in space into a matrix that the camera understands
# Original technique from
# https://gamedev.stackexchange.com/questions/168542/camera-view-matrix-from-position-yaw-pitch-worldup
# Modified similarly to: https://github.com/SuperTails/112craft 
def wsToCamMat(cp, y, p):
    yaw = -y
    
    y = yaw
    a = cp[0]
    b = cp[1]
    c = cp[2]
    
    return np.array([
        [cos(y), 0.0, -sin(y), c * sin(y) - a * cos(y)],
        [-sin(p) * sin(y), cos(p), -sin(p) * cos(y), c * sin(p) * cos(y) + a * sin(p) * sin(y) - b * cos(p)],
        [cos(p) * sin(y), sin(p), cos(p) * cos(y), -b * sin(p) - a * sin(y) * cos(p) - c * cos(y) * cos(p)],
        [0.0, 0.0, 0.0, 1.0]
    ])

# viewpoint with respect to canvas matrix
def vpToCanvasMat(vpWidth, vpHeight, canvWidth, canvHeight):
    w = canvWidth / vpWidth
    h = -canvHeight / vpHeight

    x = canvWidth * 0.5
    y = canvHeight * 0.5

    return np.array([
        [w, 0.0, x],
        [0.0, h, y],
        [0.0, 0.0, 1.0]])

# return a tuple of the distance from a point to the camera position
def wsToCam(point, cp):
    x = point[0] - cp[0]
    y = point[1] - cp[1]
    z = point[2] - cp[2]

    return [x, y, z]

# returns the distance of a camera to viewpoint in a matrix
def camToVpMat(vpDist):
    vpd = vpDist

    return np.array([
        [vpd, 0.0, 0.0, 0.0],
        [0.0, vpd, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0]
    ])
    
# camera to viewpoint
def camToVp(point, vpDist):
    # view point x and y calculated and divided based on z coord
    vpX = point[0] * vpDist / point[2]
    vpY = point[1] * vpDist / point[2]

    return [vpX, vpY]

# translate the player viewpoint from the xy point to a xy point on the canvas
def vpToCanvas(point, vpWidth, vpHeight, canvWidth, canvHeight):
    canvX = (point[0] / vpWidth + 0.5) * canvWidth
    canvY = (-point[1] / vpHeight + 0.5) * canvHeight

    return [canvX, canvY]

# convert a point in space into a xy coordinate in the canvas
def wsToCanvas(app, point):
    point = toHomogenous(point)
    mat = wsToCanvasMat(app.cameraPos, app.cameraYaw, app.cameraPitch,
        app.vpDist, app.vpWidth, app.vpHeight, app.width, app.height)

    point = mat @ point

    point = toCartesian(point)
    
    return point

# find the normal face with respect to the polygon being shown
def faceNormal(v0, v1, v2):
    v0 = toCartesian(v0)
    v1 = toCartesian(v1)
    v2 = toCartesian(v2)

    a = v1 - v0
    b = v2 - v0
    cross = np.cross(a, b)
    
    return cross

# vertices have to be within the camera space
# technique from: https://en.wikipedia.org/wiki/Back-face_culling
def isBackFace(v0, v1, v2) -> bool:
    normal = faceNormal(v0, v1, v2)
    v0 = toCartesian(v0)

    return -np.dot(v0, normal) >= 0

# check if a certain face of a block is visible. returns a bool
def blockFaceVisible(app, blockPos: BlockPos, faceIdx: int) -> bool:
    (x, y, z) = adjacentBlockPos(blockPos, faceIdx)

    if world.coordsOccupied(app, BlockPos(x, y, z)):
        return False

    return True
    
# get the light lvl of a blocks face
def blockFaceLight(app, blockPos: BlockPos, faceIdx: int) -> int:
    # get the light lvl of a block adjacent to the block in the direction of the face
    pos = adjacentBlockPos(blockPos, faceIdx)
    (chunk, (x, y, z)) = world.getChunk(app, pos)
    
    return chunk.lightLevels[x, y, z]

# returns bool saying if a back face is a back face
def isBackBlockFace(app, blockPos: BlockPos, faceIdx: int) -> bool:
    # get the face from face id, block global position
    faceIdx //= 2
    (x, y, z) = world.blockToWorld(blockPos)
    
    # distance from camera
    xDiff = app.cameraPos[0] - x
    yDiff = app.cameraPos[1] - y
    zDiff = app.cameraPos[2] - z
    
    # Left
    if faceIdx == 0:
        return xDiff > -0.5
    # Right
    elif faceIdx == 1:
        return xDiff < 0.5
    # Near
    elif faceIdx == 2:
        return zDiff > -0.5
    # Far
    elif faceIdx == 3:
        return zDiff < 0.5
    # Bottom
    elif faceIdx == 4:
        return yDiff > -0.5
    # Top
    else:
        return yDiff < 0.5

# FIXME: This doesn't conserve the CCW vertex ordering. See how he changes this later

def clip(app, vertices : List[Any], face : Face) -> List[Face]:
    # lambda function that checks whether or not face vertice is out of view
    outOfView = lambda idx: vertices[idx][2] < app.vpDist
    
    # get a numeric value holding how many faces are visible
    numVisible = (not outOfView(face[0])) + ((not outOfView(face[1])) + (not outOfView(face[2])))
    
    # if there are 0 visible faces then return empty list
    if numVisible == 0:
        return []
    # if there are 3 visible faces
    elif numVisible == 3:
        return [face]
    
    # get the xyz coordinates of each vertice of the that is in view 
    [v0, v1, v2] = sorted(face, key=outOfView)

    [[x0], [y0], [z0], _] = vertices[v0]
    [[x1], [y1], [z1], _] = vertices[v1]
    [[x2], [y2], [z2], _] = vertices[v2]

    if(numVisible == 2):
        xd = (x2 - x0) * (app.vpDist - z0) / (z2 - z0) + x0
        yd = (y2 - y0) * (app.vpDist - z0) / (z2 - z0) + y0

        xc = (x2 - x1) * (app.vpDist - z1) / (z2 - z1) + x1
        yc = (y2 - y1) * (app.vpDist - z1) / (z2 - z1) + y1

        dIdx = len(vertices)
        vertices.append(np.array([[xd], [yd], [app.vpDist], [1.0]]))
        cIdx = len(vertices)
        vertices.append(np.array([[xc], [yc], [app.vpDist], [1.0]]))

        face0 : Face = (v0, v1, dIdx)
        face1 : Face = (v0, v1, cIdx)

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

        clippedFace : Face = (v0, aIdx, bIdx)

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
def cullInstance(app, toCamMat: ndarray, inst: Instance, blockPos: Optional[BlockPos]) -> List[Tuple[Any, Face, Color]]:
    vertices = list(map(lambda v: toCamMat @ v, inst.worldSpaceVertices()))

    faces = []

    skipNext = False
    
    # go through every block and assign each block's face with its texture stored in its instance
    for (faceIdx, (face, color)) in enumerate(zip(inst.model.faces, inst.texture)):
        # if skipNext is true then we skip the block we're on and set skip to False
        if skipNext:
            skipNext = False
            continue 
        
        if blockPos is not None:
            # if block has no visible faces, ignore it
            if not inst.visibleFaces[faceIdx]:
                continue
            
            # if we see the back of a block skipNext is set to True and current block face is ignored
            if isBackBlockFace(app, blockPos, faceIdx):
                skipNext = True
                continue
            
            # if currect face is not visible skipNext is true and current face is ignored
            if not blockFaceVisible(app, blockPos, faceIdx):
                skipNext = True
                continue
            
            # if face is visible, adjust the rgb coloring of the face depending on the 
            # lightlvl it has
            light = blockFaceLight(app, blockPos, faceIdx)
            
            r = int(color[1:3], base=16)
            g = int(color[3:5], base=16)
            b = int(color[5:7], base=16)

            brightness = (light + 1) / 8
            r *= brightness
            g *= brightness
            b *= brightness
            
            r = max(0.0, min(255.0, r))
            g = max(0.0, min(255.0, g))
            b = max(0.0, min(255.0, b))

            # set new color hexcode
            color = '#{:02X}{:02X}{:02X}'.format(int(r), int(g), int(b))
        
        # if there is no block in the position we're checking and we get to the back
        # face then just skip the block
        else:
            backFace = isBackFace(
                vertices[face[0]], 
                vertices[face[1]],
                vertices[face[2]]
            )
            if backFace:
                continue

        for clippedFace in clip(app, vertices, face):
            faces.append([vertices, clippedFace, color])

    return faces

# return bool of if a block is visible or not
def blockPosIsVisible(app, pos: BlockPos) -> bool:
    pitch = app.cameraPitch
    yaw = app.cameraYaw 

    lookX = cos(pitch)*sin(-yaw)
    lookY = sin(pitch)
    lookZ = cos(pitch)*cos(-yaw)
    
    # get camPos and blockPos in respect to world
    [camX, camY, camZ] = app.cameraPos
    [blockX, blockY, blockZ] = world.blockToWorld(pos)
    
    # This is only a conservative estimate, so we move the camera "back"
    # to make sure we don't miss blocks behind us
    camX -= lookX
    camY -= lookY
    camZ -= lookZ

    dot = lookX * (blockX - camX) + lookY * (blockY - camY) + lookZ * (blockZ - camZ)

    return dot >= 0

# render instance function
def renderInstances(app, canvas):
    faces = drawToFaces(app)

    zCoord = lambda d: -(d[0][d[1][0]][2] + d[0][d[1][1]][2] + d[0][d[1][2]][2])

    faces.sort(key=zCoord)

    drawToCanvas(app, canvas, faces)

# get the faces that need to be drawn and are within the render distance set in app
def drawToFaces(app):
    toCamMat = wsToCamMat(app.cameraPos, app.cameraYaw, app.cameraPitch)
    faces = []
    for chunkPos in app.chunks:
        chunk = app.chunks[chunkPos]
        if chunk.isVisible and chunk.isFinalized:
            for (i, inst) in enumerate(chunk.instances):
                if inst is not None:
                    (inst, unburied) = inst
                    if unburied:
                        wx = chunk.pos[0] * 16 + (i // 256)
                        wy = chunk.pos[1] * 16 + (i // 16) % 16
                        wz = chunk.pos[2] * 16 + (i % 16)
                        blockPos = BlockPos(wx, wy, wz)
                        if blockPosIsVisible(app, blockPos):
                            (x, y, z) = blockPos
                            x -= app.cameraPos[0]
                            y -= app.cameraPos[1]
                            z -= app.cameraPos[2]
                            if x**2 + y**2 + z**2 <= app.renderDistanceSq:
                                faces += cullInstance(app, toCamMat, inst, blockPos)
    return faces

# draw everything to the app canvas
def drawToCanvas(app, canvas, faces):
    mat = app.csToCanvasMat

    for i in range(len(faces)):
        if type(faces[i][0]) != type((0, 0)):
            verts = list(map(lambda v: toCartesian(mat @ v), faces[i][0]))
            faces[i][0] = (verts, True)

        ((vertices, _), face, color) = faces[i]

        v0 = vertices[face[0]]
        v1 = vertices[face[1]]
        v2 = vertices[face[2]]

        if app.wireframe:
            edges = [(v0, v1), (v0, v2), (v1, v2)]

            for (v0, v1) in edges:            
                canvas.create_line(v0[0], v0[1], v1[0], v1[1], fill=color)
        else:
            canvas.create_polygon(v0[0], v0[1], v1[0], v1[1], v2[0], v2[1], fill=color)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# get a bunch of times 
frameTimes = [0.0] * 10
frameIndex = 0

def redrawAll(app, canvas):
    # get an initial time
    startTime = time.time()
    
    # creates the sky
    canvas.create_rectangle(0.0, 0.0, app.width, app.height, fill = '#0080FF')
    
    # call the renderInstance function
    renderInstances(app, canvas)
    
    # set important values and position stuffs
    origin = wsToCanvas(app, np.array([[0.0], [0.0], [0.0]]))
    xAxis = wsToCanvas(app, np.array([[1.0], [0.0], [0.0]]))
    yAxis = wsToCanvas(app, np.array([[0.0], [1.0], [0.0]]))
    zAxis = wsToCanvas(app, np.array([[0.0], [0.0], [1.0]]))

    xpoint = wsToCamMat(app.cameraPos, app.cameraYaw, app.cameraPitch) @ toHomogenous(np.array([[1.0], [0.0], [0.0]]))
    xpoint = toCartesian(xpoint)

    canvas.create_line(origin[0], origin[1], xAxis[0], xAxis[1], fill='red')
    canvas.create_line(origin[0], origin[1], yAxis[0], yAxis[1], fill='green')
    canvas.create_line(origin[0], origin[1], zAxis[0], zAxis[1], fill='blue')
    
    # cursor
    canvas.create_oval(app.width / 2 - 1, app.height / 2 - 1, app.width / 2 + 1, app.height / 2 + 1)
    
    # tick time display
    tickTime = sum(app.tickTimes) / len(app.tickTimes) * 1000.0

    # This makes it more easily legible on both dark and light backgrounds
    canvas.create_text(11, 21, text=f'Tick Time: {tickTime:.2f}ms', anchor='nw')
    canvas.create_text(10, 20, text=f'Tick Time: {tickTime:.2f}ms', anchor='nw', fill='white')
    
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
    