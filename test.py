from ursina import *

from ursina.prefabs import *

# Model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

app = Ursina()

player = Entity(model = 'cube', color = color.blue)

def update():
    player.x += held_keys['d'] * time.dt * 2
    player.x -= held_keys['a'] * time.dt * 2

    player.y += held_keys['w'] * time.dt * 2
    player.y -= held_keys['s'] * time.dt * 2
    
app.run()