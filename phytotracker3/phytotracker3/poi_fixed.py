#!/usr/bin/python3
import cv2 as cv2
import sys

def string_to_tuple(str):
    """
    Converts string to tuple for use in OpenCV
    drawing functions. Sort of an ass-backswards methond, but...

    Author: robertdcurrier@gmail.com
    Created:    2018-11-07
    Modified:   2018-11-07
    """
    color = []
    tup = map(int, str.split(','))
    for val in tup:
        color.append(val)
    color = tuple(color)
    return color


def draw_poi(frame, x, y, config,preds):
    """
    New draw_poi for phytotracker3. We lose the dashed lines
    and simply add a cross in the center. All config items
    now come from config.json

    Author: robertdcurrier@gmail.com
    Created:    2018-11-07
    Modified:   2018-11-07
    """
    if preds == 1:
        rect_color=(0,255,0)
    if preds == 0:
        rect_color=(255,0,0)

    w = config['poi']['width']
    h = config['poi']['height']
    corner_thick = config['poi']['corner_thick']
    line_thick = config['poi']['line_thick']
    dash_length = config['poi']['dash_length']
    # FAT CORNERS
    # top left
    cv2.line(frame, (x,y),(x,y+dash_length),rect_color,corner_thick)
    cv2.line(frame, (x,y),(x+dash_length,y),rect_color,corner_thick)
    # top right
    cv2.line(frame, (x+w,y),(x+w,y+dash_length),rect_color,corner_thick)
    cv2.line(frame, (x+(w-dash_length),y),(x+w,y),rect_color,corner_thick)
    # bottom left
    cv2.line(frame, (x,y+h),(x,y+(h-dash_length)),rect_color,corner_thick)
    cv2.line(frame, (x,y+h),(x+dash_length,y+h),rect_color,corner_thick)
    # bottom right
    cv2.line(frame, (x+w,y+h),(x+w,y+(h-dash_length)),rect_color,corner_thick)
    cv2.line(frame, (x+(w-dash_length),y+h),(x+w,y+h),rect_color,corner_thick)
    # Cross hairs Vertical
    cv2.line(frame, (int(x+w/2),y+3), (int(x+w/2), y+h-3), rect_color, 1)
    # Horizontal
    cv2.line(frame,(x+3,int(y+h/2)),(x+(w-3),int((y+h/2))), rect_color, 1)
