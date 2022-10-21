from cmu_112_graphics import *
import numpy as np
import math
import render
import world
from world import Chunk, ChunkPos
from typing import List
import perlin_noise

def appStarted(app):
    loadResources(app)
    
    # * World Variables and stuff~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # all the chunks
    app.chunks = {
        ChunkPos(0, 0, 0) : Chunk(ChunkPos(0, 0, 0))
    }
    
    # generate the chunk
    app.chunks[ChunkPos(0, 0, 0)].generate(app)
    
    app.timerDelay = 30
    
    app.tickTimes = [0.0] * 10
    app.tickTimeIdx = 0
    
    # * Player variables and stuff~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # player data and stuff
    app.playerHeight = 1.5
    app.playerWidth = 0.6
    app.playerRadius = app.playerWidth / 2
    app.playerOnGround = False
    app.playerVel = [0.0, 0.0, 0.0]
    app.playerWalkSpeed = 0.2
    app.playerReach = 4.0
    app.selectedBlock = 'air'
    app.gravity = 0.10
    
    # camera stuffs
    app.cameraYaw = 0
    app.cameraPitch = 0
    app.cameraPos = [4.0, 14.0 + app.playerHeight, 4.0]
    
    # * Rendering variables and stuff~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # view point informations
    app.vpDist = 0.25
    app.vpWidth = 3.0 / 4.0
    app.vpHeight = app.vpWidth * app.height / app.width
    app.wireframe = False
    app.renderDistanceSq = 6 ** 2
    
    # field of view values
    app.horiFOV = math.atan(app.vpWidth / app.vpDist)
    app.vertFOV = math.atan(app.vpHeight / app.vpDist)
    
    print(f"Horizontal FOV : {app.horiFOV} ({math.degrees(app.horiFOV)}°)")
    print(f"Vertical FOV : {app.vertFOV} ({math.degrees(app.vertFOV)}°)")
    
    # canvas to matrix thing
    app.csToCanvasMat = render.csToCanvasMat(app.vpDist, app.vpWidth, app.vpHeight, app.width, app.height)
    
    # * Input variables and stuff~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # mouse stuffs
    app.mouseMovedDelay = 10
    
    # directional key inputs
    app.w = False
    app.a = False
    app.s = False
    app.d = False
    
    app.prevMouse = None
    
    app.capMouse = False
    

def loadResources(app):
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
    faces : List[render.Face] = [
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
    
    # block models stuffs
    app.cube = render.Model(vertices, faces)

# if window size changes then we need to re-initialize the size of things in the app again
def sizeChanged(app):
    app.csToCanvasMat = render.csToCanvasMat(app.vpDist, app.vpWidth, app.vpHeight, app.width, app.height)

# define a couple things that happen when a mouse is pressed
#? Haven't added the check statements which check for a specific mouse press yet but the idea still there
def mousePressed(app, event):
    block = world.lookedAtBlock(app)
    
    if block is not None:
        (position, face) = block
        
        if(app.selectedBlock == 'air'):
            world.removeBlock(app, position)
        else:
            [x, y, z] = position
            
            if(face == 'left'):
                x -= 1
            elif(face == 'right'):
                x += 1
            elif(face == 'bottom'):
                y -= 1
            elif(face == 'top'):
                y += 1
            elif(face == 'back'):
                z -= 1
            elif(face == 'front'):
                z += 1
            
            world.addBlock(app, world.BlockPos(x, y, z), app.selectedBlock)
            
def mouseDragged(app, event):
    mouseMovedOrDragged(app, event)

# def what happens when we move the mouse across the screen.
def mouseMoved(app, event):
    mouseMovedOrDragged(app, event)
    
def mouseMovedOrDragged(app, event):
    if not app.capMouse:
        app.prevMouse = None
    
    if app.prevMouse is not None:
        xChange = -(event.x - app.prevMouse[0])
        yChange = -(event.y - app.prevMouse[1])
        
        app.cameraPitch += (yChange * .01)
        
        if(app.cameraPitch < (-math.pi / 2 * .95)):
            app.cameraPitch = (-math.pi / 2 * .95)
        elif(app.cameraPitch > (math.pi / 2 * .95)):
            app.cameraPitch > (math.pi / 2 * .95)
            
        app.cameraYaw += (xChange * .01)
        
    if app.capMouse:
        x = app.width / 2
        y = app.height / 2
        app._theRoot.event_generate('<Motion>',  warp = True, x = x, y = y)
        app.prevMouse = (x, y)

# essentially the update function will call every app.timerDelay milliseconds
def timerFired(app):
    world.tick(app)

# define what happens when keys are pressed
def keyPressed(app, event):
    if(event.key == '1'):
        app.selectedBlock = 'air'
    elif(event.key == '2'):
        app.selectedBlock = 'grass'
    elif(event.key == '3'):
        app.selectedBlock = 'stone'
    elif(event.key == 'w'):
        app.w = True
    elif(event.key == 's'):
        app.s = True
    elif(event.key == 'a'):
        app.a = True
    elif(event.key == 'd'):
        app.d = True
    elif(event.key == 'Space' and app.playerOnGround):
        app.playerVel[1] = .35
    elif(event.key == 'Escape'):
        app.capMouse = not app.capMouse
        
        if app.capMouse:
            app._theRoot.config(cursor = "none")
        else:
            app._theRoot.config(cursor = "")

# define what happens when certain keys are released
def keyReleased(app, event):
    if(event.key == 'w'):
        app.w = False
    elif(event.key == 's'):
        app.s = False
    elif(event.key == 'a'):
        app.a = False
    elif(event.key == 'd'):
        app.d = False

# call the draw function
def redrawAll(app, canvas):
    render.redrawAll(app, canvas)
    
def main():
    runApp(width = 600, height = 400, mvcCheck = False)
    
if (__name__ == '__main__'):
    main()