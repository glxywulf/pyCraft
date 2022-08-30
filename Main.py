from ursina import *

from ursina.prefabs.first_person_controller import *

from Block import *

# *Start off with the MVC model but see if you need to change things up for Ursina

# Model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

app = Ursina()

for i in range(16):
    for j in range(16):
        # for k in range(64):
        block = Block('dirt', position = (i,0,j))
        
player = FirstPersonController()

pos = str(player.position)
pos = pos[4: ]
Text.default_resolution = 1080 * Text.size
test = Text(text = pos, wordwrap = 100, origin = (9.4, -19.5)) #9.4, -19.5

def update():
    pos = str(player.position)
    pos = pos[4: ]
    test.text = pos
    
    
    
update()
app.run()

# Controller ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



# View ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~