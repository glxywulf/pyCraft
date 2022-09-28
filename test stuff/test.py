from typing import NamedTuple, List, Any, Tuple, Optional
from numpy import ndarray
import numpy as np
import heapq
# import matplotlib.pyplot as plt
from perlin_noise import *
from cmu_112_graphics import *

# class Thing(NamedTuple):
    
#     x: str # format for class instance variables 
#     y: str
    
# thing = Thing('he', 4)

# print(type(thing.y))

# ? ndarray practice stuff

# test = np.array([[1,2,3], [1,2,3]])

# print(test.shape)

# blocks: ndarray

# blocks = np.full((3,4,3), 'air') 
# # (x, y, z) = layers, rows, columns
# # (x, y) = rows, columns

# print(blocks)
# print(blocks.shape)
# print(blocks.size)

# test = [1,2,3,4,5,6]
# print(test[ :-1] / test[-1])

# cartesian : ndarray
# cartesian = np.array([[cartesian[0, 0]], [cartesian[1, 0]], [cartesian[2, 0]], [1.0]])

# assert(cartesian.shape[1] == 1)

# test = np.array([[0], [0], [0]])
# print(test)

# def isPrime(n):
#     return np.array([[7, 5], [3,-2]])

# def nthPrime(n):
#     return np.array([[4,8,2],[6,-12,1]])

# print(isPrime(9) @ nthPrime(9))

# # ? weird syntax thing

# ! @ between two ndarrays is matrix multiplication

# thing = Tuple[int, int, int]
# thing = (1,'hi',3)

# print(repr(thing[0]))
# print(f'{thing}')

# Color = str

# print(Color)
# print(thing)

# test = (1,2,3)

# print(len(test))

# thing = [None] * 20

# print(list(enumerate(thing)))

# for (i, instance) in (enumerate(thing)):
#     print((i, instance))

# noise1 = PerlinNoise(octaves=25)
# noise2 = PerlinNoise(octaves=50)
# noise3 = PerlinNoise(octaves=75)
# noise4 = PerlinNoise(octaves=100)

# width = 500
# height = 500

# pic = []

# for i in range(width):
#     row = []
#     for j in range(height):
#         noise_val = noise1([i/width, j/height])
#         noise_val += 0.5 * noise2([i/width, j/height])
#         noise_val += 0.25 * noise3([i/width, j/height])
#         noise_val += 0.125 * noise4([i/width, j/height])
        
#         row.append(noise_val)
    
#     pic.append(row)
    
# plt.imshow(pic, cmap = 'gray')
# plt.show()

# TODO Help David

# ? hasDigit, 

# def hasProperty309(n):
#     n = n ** 5
    
#     has1 = False
#     has2 = False
#     has3 = False
#     has4 = False
#     has5 = False
#     has6 = False
#     has7 = False
#     has8 = False
#     has9 = False
#     has0 = False
    
#     while(n > 0):
#         check = n % 10
        
#         if(check == 0):
#             has0 = True
#         elif(check == 1):
#             has1 = True
#         elif(check == 2):
#             has2 = True
#         elif(check == 3):
#             has3 = True
#         elif(check == 4):
#             has4 = True
#         elif(check == 5):
#             has5 = True
#         elif(check == 6):
#             has6 = True
#         elif(check == 7):
#             has7 = True
#         elif(check == 8):
#             has8 = True
#         elif(check == 9):
#             has9 = True
            
#         n //= 10
        
#     if not(has0 and has1 and has2 and has3 and has4 and has5 and has6 and has7 and has8 and has9):
#         return False
    
#     return True
        
    
# print(hasProperty309(418))


# ? ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# faces = [
#         # Left
#         (0, 2, 1),
#         (1, 2, 3),
#         # Right
#         (4, 5, 6),
#         (6, 5, 7),
#         # Near
#         (0, 4, 2),
#         (2, 4, 6),
#         # Far
#         (5, 1, 3),
#         (5, 3, 7),
#         # Bottom
#         (0, 1, 4),
#         (4, 1, 5),
#         # Top
#         (3, 2, 6),
#         (3, 6, 7),
#     ]

# # (np.array([[modelX], [modelY], [modelZ]]), texture)

# def listOfVert():
#     # result variable list
#     result = []
    
#     # for every vertice add the translation value and then append into result list
#     for vertice in faces:
#         result.append(vertice + 2)
    
#     return result

# def thing(a):
#     assert(a.shape[1] == 1)
    
#     return np.array([[a[0, 0]], [a[1, 0]], [a[2, 0]], [1]])

# print(listOfVert())

# a = np.array([[1], [2], [3], [4]])

# print(a)
# print(a.ravel())
# print(a[ :-1] / a[-1])

# 3/4, 3/4, 1000, 1000

# w = 1000 / (3/4)
# h = -1000 / (3/4)

# x = 1000 * 0.5
# y = 1000 * 0.5

# vptcm = np.array([
#     [w, 0, x],
#     [0, h, y],
#     [0, 0, 1]
# ])

# vpd = .25

# ctvm = np.array([
#     [vpd, 0, 0, 0],
#     [0, vpd, 0, 0],
#     [0, 0, 1, 0]
# ])

# cp = [4.0, 10.0 + 1.5, 4.0]
# y = 0

# print(np.array([[-1.0], [-1.0], [-1.0]]) / 2.0)

# blocks = np.full((16,16,16), 'air')
# # print(blocks.shape)

# def coordToID(position):
#         (x, y, z) = position
#         (xB, yB, zB) = blocks.shape
        
#         return x * 16 * 16 + y * 16 + z
        
# def coordFromID(id, thing):
#     x, y, z = thing

#     xID = id // (y * z)
#     yID = (id // z) % y
#     zID = id % z
    
#     return (xID, yID, zID)


# print(coordToID((15, 15, 15)))
# print(coordFromID(4095))


# def thing(t):
#     for i in range(t):
#         yield i
        
# print(list(thing(10))[1])

# thing = [(-1, 0, 0), (1, 0, 0), (0, 0, -1), (0, 0, 1), (0, -1, 0), (0, 1, 0)][1]

# print(thing)

# instances = [None] * (64)
# print(instances)

# blocks = np.full((3, 3, 3), 'air')

# for x in range(3):
#     for y in range(3):
#         for z in range(3):
#             blocks[x, y, z] = x + y + z
            
# print(blocks)

# print(blocks[2, 2, 2])

# test = np.full((16,16,16), 0)

# print(test.shape)

# test = [3,6,2,7,2,4,8,1,-9,9,0]

# heapq.heappop(test)

# print(heapq.heappop(test))

def appStarted(app):
    app.cx = app.width // 2
    app.cy = app.height // 2
    
    app.timerDelay = 10
    
    app.dx = 5
    app.dy = 3
    
def timerFired(app):
    bounce(app)
    app.cx += app.dx
    app.cy += app.dy
    
def bounce(app):
    if(app.cx - 50 < 0 or app.cx + 50 > app.width):
        app.dx *= -1
    if(app.cy - 50 < 0 or app.cy + 50 > app.height):
        app.dy *= -1
    
def redrawAll(app, canvas):
    canvas.create_rectangle(app.cx - 50, app.cy - 50, app.cx + 50, app.cy + 50, fill = 'blue', width = 0)

runApp(width = 500, height = 500)