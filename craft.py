from cmu_112_graphics import *
import numpy as np
import math
import render
import world
from world import Chunk, ChunkPosition
import perlin_noise

def appStarted(app):
    # vertices
    vertices = [
        np.array([[-1], [-1], [-1]]) / 2,
        np.array([[-1], [-1], [1]]) / 2,
        np.array([[-1], [1], [-1]]) / 2,
        np.array([[-1], [1], [1]]) / 2,
        np.array([[1], [-1], [-1]]) / 2,
        np.array([[1], [-1], [1]]) / 2,
        np.array([[1], [1], [-1]]) / 2,
        np.array([[1], [1], [1]]) / 2
    ]
    
    # block textures, currently just hex code colors
    grassTexture = [
        '#FF0000', '#FF0000',
        '#A52A2A', '#A52A2A',
        '#A52A2A', '#A52A2A',
        '#A52A2A', '#A52A2A',
        '#A52A2A', '#A52A2A',
        '#00FF00', '#00EE00']
    
    stoneTexture = [
        '#AAAAAA', '#AAAABB',
        '#AAAACC', '#AABBBB',
        '#AACCCC', '#88AAAA',
        '#AA88AA', '#888888',
        '#AA88CC', '#778888',
        '#BBCCAA', '#BBBBBB'
    ]
    
    # texture dictionary for efficient calling
    app.textures = {
        'grass' : grassTexture,
        'stone' : stoneTexture
    }
    
    # tuples that contain directional values equivalent to faces
    faces = [
        # Left
        (0, 2, 1),
        (1, 2, 3),
        # Right
        (4, 5, 6),
        (6, 5, 7),
        # Front
        (0, 4, 2),
        (2, 4, 6),
        # Back
        (5, 1, 3),
        (5, 3, 7),
        # Bottom
        (0, 1, 4),
        (4, 1, 5),
        # Top
        (3, 2, 6),
        (3, 6, 7),
    ]
    
    # terrain variation scale
    app.lowNoise = perlin_noise.PerlinNoise(octaves = 3)
    
    # block models stuffs
    app.cube = render.Model(vertices, faces)
    
    # all the chunks
    app.chunks = {
        ChunkPosition(0, 0, 0) : Chunk(ChunkPosition(0, 0, 0))
    }
    
    # generate the chunk
    app.chunks[ChunkPosition(0, 0, 0)].generate(app)
    
    # player data and stuff
    app.playerHeight = 1.5
    app.playerWidth = 0.6
    app.playerRadius = app.playerWidth / 2
    app.onGround = False
    app.playerVelocity = [0.0, 0.0, 0.0]
    app.walkSpeed = 0.2
    app.selectedBlock = 'air'
    app.gravity = 0.10
    app.renderDistSq = 6**2
    
    # camera stuffs
    app.camYaw = 0
    app.camPitch = 0
    app.camPos = [4.0, 10.0 + app.playerHeight, 4.0]
    
    # view point informations
    app.vpDist = 0.25
    app.vpWidth = 3.0 / 4.0
    app.vpHeight = app.vpWidth * app.height / app.width
    
    # field of view values
    app.horiFOV = math.atan(app.vpWidth / app.vpDist)
    app.vertFOV = math.atan(app.vpHeight / app.vpDist)
    
    print(f"Horizontal FOV : {app.horiFOV} ({math.degrees(app.horiFOV)}°)")
    
    app.timerDelay = 50
    
    # directional key inputs
    app.w = False
    app.a = False
    app.s = False
    app.d = False
    
    # mouse stuffs
    app.prevMouseInput = None
    
    app.capMouse = False
    
    # block stuffs
    app.wireFrame = False
    
    # canvas to matrix thing
    app.csToCanvMat = render.csToCanvasMat(app.vpDist, app.vpWidth, app.vpHeight, app.width, app.height)
    
def sizeChanged(app):
    app.csToCanvMat = render.csToCanvasMat(app.vpDist, app.vpWidth, app.vpHeight, app.width, app.height)

def mousePressed(app, event):
    block = world.lookBlock(app)

def redrawAll(app, canvas):
    render.redrawAll(app, canvas)
    
runApp(width = 1000, height = 1000, mvcCheck = False)