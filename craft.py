from cmu_112_graphics import *
import numpy as np
import math
import render
import world
# from world import Chunk, ChunkPos
from typing import List
import perlin_noise

def appStarted(app):
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
    
    app.textures = {
        'grass' : grassTexture,
        'stone' : stoneTexture
    }
    
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
    
    app.lowNoise = perlin_noise.PerlinNoise(octaves = 3)
    
    app.cube = render.Model(vertices, faces)

def redrawAll(app, canvas):
    render.redrawAll(app, canvas)
    
runApp(width = 1000, height = 1000, mvcCheck = False)