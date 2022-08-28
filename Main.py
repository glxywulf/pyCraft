from ursina import *

from ursina.prefabs.first_person_controller import *

from Block import *

# *Start off with the MVC model but see if you need to change things up for Ursina

# Model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

app = Ursina()

for i in range(16):
    for j in range(16):
        block = Block(position = (i,0,j))
        
player = FirstPersonController()


app.run()

# Controller ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# View ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~