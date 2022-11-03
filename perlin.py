import random, math

# Vectors from https://mrl.cs.nyu.edu/~perlin/paper445.pdf
def getPerlinVectors(x, y, width):
    
    gridX = math.floor(x / width)
    gridY = math.floor(y / width)
    
    vectors = [(1,1,0), (-1,1,0), (1,-1,0), (-1,-1,0), (1,0,1), (-1,0,1), (1,0,-1), (-1,0,-1), (0,1,1), (0,-1,1), (0,1,-1), (0,-1,-1)]
    
    combine = lambda gx, gy : gx * 10_0000 + gy
    
    # top-left
    random.seed(combine(gridX, gridY))
    vtl = random.choice(vectors)
    
    # top-right
    random.seed(combine(gridX + 1, gridY))
    vtr = random.choice(vectors)
    
    # bottom-left
    random.seed(combine(gridX, gridY + 1))
    vbl = random.choice(vectors)
    
    # bottom-right
    random.seed(combine(gridX + 1, gridY + 1))
    vbr = random.choice(vectors)
    
    return (vtl, vtr, vbl, vbr)
    
# Using explanations from https://adrianb.io/2014/08/09/perlinnoise.html
# (it's all written in heavily-optimized C so no code was used)
def getPerlinValue(x, y, width):
    (vtl, vtr, vbl, vbr) = getPerlinVectors(x, y, width)

    gridX = math.floor(x / width)
    gridY = math.floor(y / width)

    x -= gridX * width
    y -= gridY * width

    x /= width
    y /= width
    
    # top-left
    dtl = vtl[0] * x + vtl[1] * y
    # top-right
    dtr = vtr[0] * (x-1.0) + vtr[1] * y
    # bottom-left
    dbl = vbl[0] * x + vbl[1] * (y-1.0)
    # bottom-right
    dbr = vbr[0] * (x-1.0) + vbr[1] * (y-1.0)
    
    # lambda functions to get slope and intercept of values
    slope = lambda t : 6 * t**5 - 15 * t**4 + 10 * t**3
    interp = lambda a, b, t : b * slope(t) + (1.0 - slope(t)) * a
    
    top = interp(dtl, dtr, x)
    bot = interp(dbl, dbr, x)

    return interp(top, bot, y)

def timerFired(app):
    app.xOff += 20
    pass

# This returns stuff *probably* in the range -0.5 to 0.5.
# No promises.
def getPerlinFractal(x, y, baseFreq, octaves):
    freq = baseFreq
    total = 0.0
    totalscale = 0.0
    persist = 0.5
    
    for i in range(octaves):
        freq *= 2
        amplitude = 0.5**i

        total += amplitude * getPerlinValue(x, y, 1.0 / freq)
        totalscale += amplitude

    return (total / totalscale)