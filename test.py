from ursina import *

from ursina.prefabs import *

from Block import *

# Model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

app = Ursina()

block = Entity(model = 'cube', color = color.blue, texture = 'white_cube')

button = Button(parent = scene, model = 'cube', texture = 'brick', color = color.white, highlight_color = color.green, pressed_color = color.red)

def update():
    block.x += held_keys['d'] * time.dt * 2
    block.x -= held_keys['a'] * time.dt * 2

    block.y += held_keys['w'] * time.dt * 2
    block.y -= held_keys['s'] * time.dt * 2
    
app.run()