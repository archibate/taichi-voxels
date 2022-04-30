from scene import Scene
import taichi as ti
from taichi.math import *
import numpy as np


def load_font():
    arr = np.empty((256, 16), dtype=np.uint8)
    with open('hankaku.txt', 'r') as f:
        lines = f.readlines()
        for i in range(256):
            for j in range(16):
                line = lines[18 * i + 3 + j]
                b = 0
                for k in range(8):
                    if line[k] == '*':
                        b |= 1 << k
                arr[i][j] = b
    return arr


scene = Scene(exposure=10)
scene.set_floor(0.0, (1.0, 1.0, 1.0))
scene.set_background_color((0, 0, 0))

font = ti.field(ti.u8, (256, 16))
text = ti.field(ti.u8, 8192)
text_len = ti.field(ti.i32, ())


def set_text_array(s):
    for i in range(len(s)):
        text[i] = ord(s[i])
    text_len[None] = len(s)


WHITE = vec3(0.9, 0.9, 0.9)
RED = vec3(0.9, 0.1, 0.1)
BLUE = vec3(0.1, 0.1, 0.9)


@ti.func
def font_at(c, i, j):
    return 1 if (font[c, 15 - j] & (1 << i)) != 0 else 0


@ti.func
def paint_font(c, x, y):
    for i, j in ti.ndrange(8, 16):
        scene.set_voxel(vec3(x * 8 + i, y * 16 + j, 0), font_at(c, i, j), RED if x % 2 == 0 else BLUE)


@ti.kernel
def initialize_voxels():
    for i, j in ti.ndrange((-20, 20), (-20, 20)):
        scene.set_voxel(vec3(i, 60, j), 2, WHITE)


@ti.kernel
def update_voxels():
    for i in range(text_len[None]):
        c = ti.cast(text[i], ti.i32)
        paint_font(c, i - text_len[None] // 2, 0)


font.from_numpy(load_font())
initialize_voxels()
set_text_array('Hello Taichi!')
update_voxels()

def callback(window):
    if window.is_pressed('o'):
        set_text_array(input('Please input your text: '))
        scene.clear_voxels()
        initialize_voxels()
        update_voxels()
        return True
    return False

scene.finish(callback)
