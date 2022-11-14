from cmu_112_graphics import *
import numpy as np
import math
import render
import world
from button import Button, ButtonManager
from world import Chunk, ChunkPos
from typing import List
from enum import Enum

class GameState(Enum):
    STARTUP = 1
    TITLE = 2
    PLAYING = 3

def createSizedBackground(app, width : int, height : int):
    cobble = app.loadImage('assets/cobbleBackground.jpg')
    cobble = app.scaleImage(cobble, 2)
    cWidth, cHeight = cobble.size
    
    newCobble = Image.new(cobble.mode, (width, height))
    
    for xIdx in range(math.ceil(width / cWidth)):
        for yIdx in range(math.ceil(height / cHeight)):
            xOffset = xIdx * cWidth
            yOffset = yIdx * cHeight
            
            newCobble.paste(cobble, (xOffset, yOffset))
            
    return newCobble

def appStarted(app):
    loadResources(app)
    
    app.titleText = app.loadImage('assets/titleText.png')
    app.titleText = app.scaleImage(app.titleText, 3)
    
    app.btnBg = createSizedBackground(app, 200, 40)
    
    app.state = GameState.TITLE
    
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
    app.cameraPos = [4.0, 8.0 + app.playerHeight, 4.0]
    
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
    
    app.captureMouse = False
    
    app.buttons = ButtonManager()
    
    # FIXME: This does not work with resizing!
    # type: ignore
    app.buttons.addButton('playSurvival', Button(app.width / 2, app.height / 2, background = app.btnBg, text = "Play Survival")) 
    

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
    
    leavesTexture = [
        '#206000', '#256505',
        '#257000', '#256505',
        '#206010', '#206505',
        '#206505', '#256005',
        '#306005', '#256500',
        '#206500', '#306505',
    ]
    
    logTexture = [
        '#705020', '#655020',
        '#705520', '#655025',
        '#705025', '#705020',
        '#755020', '#705A2A',
        '#755520', '#7A4A20',
        '#705525', '#70502A',
    ]
    
    # texture dictionary for efficient calling
    app.textures = {
        'grass' : grassTexture,
        'stone' : stoneTexture,
        'leaves' : leavesTexture,
        'log' : logTexture
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
    app.buttons.onPress(event.x, event.y)
    
    block = world.lookedAtBlock(app)
    
    if block is not None:
        (pos, face) = block
        
        if(app.selectedBlock == 'air'):
            world.removeBlock(app, pos)
        else:
            [x, y, z] = pos
            
            if (face == 'left'):
                x -= 1
            elif (face == 'right'):
                x += 1
            elif (face == 'bottom'):
                y -= 1
            elif (face == 'top'):
                y += 1
            elif (face == 'back'):
                z -= 1
            elif (face == 'front'):
                z += 1
                            
            world.addBlock(app, world.BlockPos(x, y, z), app.selectedBlock)
            
def mouseReleased(app, event):
    btn = app.buttons.onRelease(event.x, event.y)
    
    if(btn is not None):
        print(f"Pressed{btn}")
        
        if(btn == 'playSurvival'):
            app.state = GameState.PLAYING
            app.buttons.buttons = {}
            
def mouseDragged(app, event):
    mouseMovedOrDragged(app, event)

# def what happens when we move the mouse across the screen.
def mouseMoved(app, event):
    mouseMovedOrDragged(app, event)
    
def mouseMovedOrDragged(app, event):
    if not app.captureMouse:
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
        
    if app.captureMouse:
        x = app.width / 2
        y = app.height / 2
        app._theRoot.event_generate('<Motion>',  warp = True, x = x, y = y)
        app.prevMouse = (x, y)

# essentially the update function will call every app.timerDelay milliseconds
def timerFired(app):
    if(app.state == GameState.TITLE):
        app.cameraYaw += 0.01
    
    world.tick(app)

# define what happens when keys are pressed
def keyPressed(app, event):
    if(app.state == GameState.TITLE):
        return
    
    if(event.key == '1'):
        app.selectedBlock = 'air'
    elif(event.key == '2'):
        app.selectedBlock = 'grass'
    elif(event.key == '3'):
        app.selectedBlock = 'stone'
    elif(event.key == '4'):
        app.selectedBlock = 'leaves'
    elif(event.key == '5'):
        app.selectedBlock = 'log'
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
        app.captureMouse = not app.captureMouse
        
        if app.captureMouse:
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
        
def onPlayClicked(app):
    print("foobar")
    app.state = GameState.PLAYING
    
# From:
# https://www.kosbie.net/cmu/fall-19/15-112/notes/notes-animations-part2.html
def getCachedImage(image):
    if ('cachedPhotoImage' not in image.__dict__):
        image.cachedPhotoImage = ImageTk.PhotoImage(image)
    return image.cachedPhotoImage

# call the draw function
def redrawAll(app, canvas):
    render.redrawAll(app, canvas)
    
    if app.state == GameState.TITLE:
        canvas.create_image(app.width / 2, 50, image=getCachedImage(app.titleText))
    
    app.buttons.draw(app, canvas)
    
def main():
    runApp(width = 600, height = 400, mvcCheck = True)
    
if (__name__ == '__main__'):
    main()