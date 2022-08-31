from ursina import *

class Block(Button):
    def __init__(self, blockType, position = (0,0,0)):
        if(blockType != None):
            super().__init__(parent = scene, 
                            position = position, 
                            model = 'cube', 
                            origin_y = 0.5, 
                            texture = load_texture(f'assets\{blockType}'), 
                            highlight_color = color.green, 
                            color = color.white)
        else:
            super().__init__(parent = scene, 
                    position = position, 
                    model = 'cube', 
                    origin_y = 0.5, 
                    texture = 'white_cube', 
                    highlight_color = color.green, 
                    color = color.white)

        self.blockType = blockType
        
        
    def input(self, key):
        if(self.hovered):
            if(key == 'left mouse down'):
                print(self.blockType)
                destroy(self)
            if(key == 'right mouse down'):
                print(self.blockType)
                block = Block('dirt', position = self.position + mouse.normal)