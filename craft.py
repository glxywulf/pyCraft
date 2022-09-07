from cmu_112_graphics import *
import numpy as np
import math
import render
import world
# from world import Chunk, ChunkPos
from typing import List
import perlin_noise

def redrawAll(app, canvas):
    render.redrawAll(app, canvas)
    
runApp(width = 1000, height = 1000, mvcCheck = False)