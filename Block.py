from ursina import *

class Block(Button):
    def __init__(self, position = (0,0,0)):
        super().__init__(parent = scene, position = position, model = 'cube', origin_y = 0.5, texture = 'white_cube', highlight_color = color.green, color = color.white)
        
    def input(self, key):
        if(self.hovered):
            if(key == 'left mouse down'):
                destroy(self)
            if(key == 'right mouse down'):
                block = Block(position = self.position + mouse.normal)