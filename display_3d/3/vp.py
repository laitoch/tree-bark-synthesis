#!/usr/bin/env python3

from vpython import *
# from scipy.ndimage import imread

file = './bi.png'
bumpmap = './bumpmap.png'

# w,h = imread(filename, mode='L').shape
# print(w,h)
w,h = (800,200)

scene.range = 1
scene.forward = vector(0,5,10)
scene.height = 2000
scene.width = 1600
scene.ambient = color.gray(0.3)
lamp = local_light(pos=vector(4,14,.5), color=color.gray(0.4))

# cylinder(pos=vector(0,2+2.54,5), up=vector(1,0,0), radius=1.28, length=5.08, texture=filename)
cylinder(pos=vector(0,2.9+4,5), up=vector(1,0,0), radius=2/3.14, length=8,
        texture={'file':file, 'bumpmap':bumpmap, 'turn':1})



s = '\nThis illustrates the use of an image from another web site as a texture.\n'
s += 'This is an example of CORS, "Cross-Origin Resource Sharing".\n'
scene.caption = s
scene.append_to_caption("""
Right button drag or Ctrl-drag to rotate "camera" to view scene.
To zoom, drag with middle button or Alt/Option depressed, or use scroll wheel.
  On a two-button mouse, middle is left + right.
Shift-drag to pan left/right and up/down.
Touch screen: pinch/extend to zoom, swipe or two-finger rotate.""")
