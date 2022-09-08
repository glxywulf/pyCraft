import math
import time
import world
import numpy as np
from cmu_112_graphics import *
from math import sin, cos
from numpy import ndarray
from typing import List, Tuple, Optional, Any
# from world import BlockPos, adjacentBlockPos

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
        self.vertices = list(map(thing, self.listOfVert())) # ! Figure out what this is
        
    # returns list of translated vertices    
    def listOfVert(self):
        # result variable list
        result = []
        
        # for every vertice add the translation value and then append into result list
        for vertice in self.model.vertices:
            result.append(vertice + self.trans)
        
        return result
    
# ! Figure out what this is
def thing(a):
    assert(a.shape[1] == 1)
    
    return np.array([[a[0, 0]], [a[1, 0]], [a[2, 0]], [1]])
    
    


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def drawToCanvas(app, canvas, faces):
    pass

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
