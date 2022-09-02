from typing import NamedTuple, List, Any, Tuple, Optional
from numpy import ndarray
import numpy as np

class Thing(NamedTuple):
    
    x: str # format for class instance variables 
    y: str
    
thing = Thing('he', 4)

# print(type(thing.y))

test = np.array([[1,2,3], [1,2,3]])

print(test.shape)



from cmu_112_graphics import *
import numpy as np
import math
import heapq
import time
from collections import namedtuple
from collections import deque
from math import cos, sin
from numpy import infty, ndarray
from typing import Optional, NamedTuple, List, Tuple, Any, Union
import copy
import perlin_noise

def appStarted(app):
    vertices = [
        np.array([[-1.0], [-1.0], [-1.0]]) / 2.0,
        np.array([[-1.0], [-1.0], [1.0]]) / 2.0,
        np.array([[-1.0], [1.0], [-1.0]]) / 2.0,
        np.array([[-1.0], [1.0], [1.0]]) / 2.0,
        np.array([[1.0], [-1.0], [-1.0]]) / 2.0,
        np.array([[1.0], [-1.0], [1.0]]) / 2.0,
        np.array([[1.0], [1.0], [-1.0]]) / 2.0,
        np.array([[1.0], [1.0], [1.0]]) / 2.0
    ]
    
    app.csToCanvasMat = csToCanvasMat(app.vpDist, app.vpWidth, app.vpHeight, app.width, app.height)
    
    app.vpDist = 0.25
    app.vpWidth = 3.0 / 4.0
    app.vpHeight = app.vpWidth * app.height / app.width 
    
def sizeChanged(app):
    app.csToCanvasMat = csToCanvasMat(app.vpDist, app.vpWidth, app.vpHeight, app.width, app.height)
    
def csToCanvasMat(vpDist, vpWidth, vpHeight, canvWidth, canvHeight):
    vpToCanv = vpToCanvasMat(vpWidth, vpHeight, canvWidth, canvHeight)
    return vpToCanv @ camToVpMat(vpDist)

def vpToCanvasMat(vpWidth, vpHeight, canvWidth, canvHeight):
    w = canvWidth / vpWidth
    h = -canvHeight / vpHeight

    x = canvWidth * 0.5
    y = canvHeight * 0.5

    return np.array([
        [w, 0.0, x],
        [0.0, h, y],
        [0.0, 0.0, 1.0]])
    
def camToVpMat(vpDist):
    vpd = vpDist

    return np.array([
        [vpd, 0.0, 0.0, 0.0],
        [0.0, vpd, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0]
    ])
    
runApp(width = 1280, height = 720)