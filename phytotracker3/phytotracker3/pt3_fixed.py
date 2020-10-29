#!/usr/bin/env python3
# coding: utf-8
"""
This is the PlaidML version of Phytotracker3
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
# Now using PlaidML
import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import keras
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import img_to_array

# Helper libraries
import json
import time
import sys
import argparse
import numpy as np
import cv2 as cv2
import poi_fixed as poi

def show_info():
    """
    Prints info when called from command line w/no options

    Author: robertdcurrier@gmail.com
    Created:    2018-11-06
    Modified:   2018-11-06
    """
    print("show_info(): No input file specified.")
    config = get_config()
    print(config['captions']['title'])
    print("Testing load_model()")
    load_model()
    print("Testing load_scale()")
    load_scale()
    # Goombye
    sys.exit()


def get_config():
    """
    Get config settings from config.json. We no longer use configparser

    Author: robertdcurrier@gmail.com
    Created:    2019-07-18
    Modified:   2019-07-18
    """
    c_file = open('configs/pt3_fixed.cfg').read()
    config = json.loads(c_file)
    return config


def load_model():
    """
    Loads TensorFlow model and cell weights from lib
    Author: robertdcurrier@gmail.com
    Created:    2019-07-18
    Modified:   2018-07-18
    """
    config = get_config()
    args = get_args()

    if args["taxa"]:
        taxa = args["taxa"]
        model_file = config['taxa'][taxa]['model_json']
    else:
        taxa = config['system']['default_taxa']
        model_file = config['taxa'][taxa]['model_json']

    print("Using %s as taxa and %s as model_file" % (taxa, model_file))

    try:
        json_file = open(model_file)
        print("load_model(): Loaded %s model_file successfully" % taxa)
    except IOError:
        print("load_model(): Failed to open %s model_file"  % model_file)
        sys.exit()

    model_json = json_file.read()
    json_file.close()
    model = keras.models.model_from_json(model_json)
    # load weights into new model
    if args["taxa"]:
        taxa = args["taxa"]
        weights_file = config['taxa'][taxa]['weights_file']
    else:
        taxa = config['system']['default_taxa']
        weights_file = config['taxa'][taxa]['weights_file']

    try:
        model.load_weights(weights_file)
    except IOError:
        print("load_model(): Failed to open %s" % weights_file)
        sys.exit()
    return model


def load_scale():
    """
    Loads scale file from lib

    Author: robertdcurrier@gmail.com
    Created:    2018-11-06
    Modified:   2018-11-06
    """
    config = get_config()
    args = get_args()
    if args["taxa"]:
        taxa = args["taxa"]
        scale_file = config['taxa'][taxa]['scale_file']
    else:
        taxa = config['system']['default_taxa']
        scale_file = config['taxa'][taxa]['scale_file']

    try:
        scale_file = open(scale_file)
    except IOError:
        print("load_scale(): Failed to open %s"  % scale_file)
        sys.exit()

    scale = json.loads(scale_file.read())
    scale_file.close()
    print("Loaded %s scale successfully" % taxa)
    return scale


def get_args():
    """
    Gets command line args

    Author: robertdcurrier@gmail.com
    Created:    2018-11-06
    Modified:   2019-11-06
    """
    arg_p = argparse.ArgumentParser()
    arg_p.add_argument("-i", "--input",
                       help="Path to the input image file")
    arg_p.add_argument("-o", "--output",
                       help="Path to the output image file")
    arg_p.add_argument("-g", "--gui", help="GUI: for testing on desktops",
                       action="store_true")
    arg_p.add_argument("-z", "--zoom", help="zoom factor")
    arg_p.add_argument("-r", "--report", help="report MAX_CELLS to STDOUT",
                       action="store_true")
    arg_p.add_argument("-l", "--learning", help="generate training sets",
                       action="store_true")
    arg_p.add_argument("-t", "--taxa", help="choose taxa")
    arg_p.add_argument("-c", "--contours", help="show contours",
                       action="store_true")
    arg_p.add_argument("-p", "--poi", help="show target tracking reticule",
                       action="store_true")

    args = vars(arg_p.parse_args())
    return args


def validate_taxa(args):
    """
    If --taxa arg is used make sure we have this taxa
    in the config file
    """
    config = get_config()
    taxa = args["taxa"]
    print("validate_taxa(): Validating %s" % taxa)
    if taxa in config['taxa'].keys():
        print("validate_taxa(): Found %s taxa settings" % taxa)
    else:
        print("validate_taxa(): %s not found!" % taxa)
        sys.exit()


def process_image(file_name, model, taxa):
    """
    Main body of app. Reads image input file, gets cons list,
    draws target indicator and updates settings. Skips first n images
    and last n images to avoid shaking. N defined in config file

    Author: robertdcurrier@gmail.com
    Created:    2018-11-06
    Modified:   2018-11-06
    """
    args = get_args()

    config = get_config()
    scale = load_scale()
    print("process_image(): Using taxa %s" % taxa)

    try:
        image = cv2.imread(file_name)
    except IOError as e:
        print("process_image(): Failed to open %s. %s" % (file_name, e))
        sys.exit()

    thresh_min = config['taxa'][taxa]['thresh_min']
    thresh_max = config['taxa'][taxa]['thresh_max']
    con_min = config['taxa'][taxa]['con_min']
    con_max = config['taxa'][taxa]['con_max']
    if config['system']['debug']:
        print("process_image(): loading %s" % file_name)
        print("process_image(): thresh_min %d" % thresh_min)
        print("process_image(): thresh_max %d" % thresh_max)
        print("process_image(): con_min %d" % con_min)
        print("process_image(): con_max %d" % con_max)

    # Equalize histogram and get contours
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, thresh_min, thresh_max, 0)
    edges = cv2.Canny(thresh, 100, 200)

    contours, _ = (cv2.findContours(thresh, cv2.RETR_TREE,
                                    cv2.CHAIN_APPROX_SIMPLE))

    good_cons = gen_contours(contours, config, taxa)
    for con in good_cons:
        (x, y), radius = cv2.minEnclosingCircle(con)

    if args["contours"]:
        print("process_image(): Drawing %d contours" % len(good_cons))
        color = (0, 0, 255)
        thick = 1
        if not args["learning"]:
            cv2.drawContours(image, good_cons, -1, color, thick)


    if args["poi"]:
        target_cells(image, good_cons, config, model)

    #zoom or shrink
    if args["zoom"]:
        gray = scale_image(gray.copy(), args)

    if args["learning"]:
        print('Snipping cells...')
        snip_cells(image,good_cons)

    if args["gui"]:
        # Show image
        show_image(image)


def snip_cells(image, contours):
    """
    Cuts out contours for training
    """
    print("snip_cells(): Got %d contours" % len(contours))
    print("snip_cells(): Cutting out ROIs")
    for con in contours:
        (x_pos, y_pos, w, h) = cv2.boundingRect(con)
        roi = (image[y_pos:y_pos+h, x_pos:x_pos+w])
        if roi.size != 0:
            roi = cv2.resize(roi, (64, 64))
            epoch = int(time.time()*10000)
            img_name = ('/Volumes/data/phyto3/tmp/%d.jpg' % epoch)
            if len(roi) == 64:
                print("Writing image %s" % img_name)
                cv2.imwrite(img_name, roi)
        else:
            print("Invalid ROI, not snipping...")
    # Wait until end to draw bounding boxes so we don't get bleedover
    for con in contours:
        (x_pos, y_pos, w, h) = cv2.boundingRect(con)
        start_pos = (x_pos, y_pos)
        end_pos = (x_pos+w, y_pos+h)
        thickness = 1
        color = (0,0,255)
        cv2.rectangle(image, start_pos, end_pos, color, thickness)


def gen_contours(contours, config, taxa):
    """
    Generates contour list for each image based on config settings

    Author: robertdcurrier@gmail.com
    Created:    2018-11-06
    Modified:   2018-11-07
    """
    print("gen_contours(): looking for good cons...")
    good_cons = []
    con_min = config['taxa'][taxa]['con_min']
    con_max = config['taxa'][taxa]['con_max']
    radius_min = config['taxa'][taxa]['radius_min']
    radius_max = config['taxa'][taxa]['radius_max']

    for con in contours:
        (x, y), radius = cv2.minEnclosingCircle(con)
        if cv2.contourArea(con) > con_min and cv2.contourArea(con) < con_max:
            if radius > radius_min and radius < radius_max:
                good_cons.append(con)
    return good_cons


def classify_cell(model, cell):
    """
    Classifies cells

    Author: robertdcurrier@gmail.com
    Created:    2018-11-06
    Modified:   2019-04-05

    We changed this up a bit... we are now passing buffers,
    not writing to file for cell images. We reshape to
    64, 64 to match the model. If mode.predict returns none for
    some reason we return a 2 so we can ignore or flag. Brevis
    returns a 0 and Not Brevis returns a 1.
    """

    brevis = False
    if len(cell) == 40:
        cell = cv2.resize(cell, (40, 40))
        x = img_to_array(cell)
        x = x.reshape((1,) + x.shape)
        x = preprocess_input(x)
        preds = (model.predict_classes(x, verbose=0))
        if preds is None:
            return 2
        if preds == 0:
            return 0
        if preds == 1:
            return 1



def target_cells(image, good_cons, config, model):
    """
    Draws 'Person of Interest' target around detected cells
    if TensorFlow model returns True. If False then draw
    a red rectangle if '-a' option set.

    Author: robertdcurrier@gmail.com
    Created:    2018-11-06
    Modified:   2018-11-07
    """
    print("target_cells()")

    taxa = config['system']['default_taxa']
    x_offset = config['taxa'][taxa]['x_offset']
    y_offset = config['taxa'][taxa]['y_offset']
    roi_offset = config['taxa'][taxa]['roi_offset']
    preds = 0
    cell_count = 0
    args = get_args()

    for con in good_cons:
        (x_pos, y_pos, w, h) = cv2.boundingRect(con)
        cx = 0
        cy = 0
        M = cv2.moments(con)
        if int(M['m10']) > 0 and int(M['m01']) > 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            roi = (image[cy-roi_offset:cy+roi_offset,
                   cx-roi_offset:cx+roi_offset])
            if roi.size != 0:
                roi = cv2.resize(roi, (40, 40))
            epoch = int(time.time()*1000)
            if args['learning']:
                # GET THIS IN CONFIG.JSON YO
                img_name = ('/data/phyto3/tmp/%d.jpg' % epoch)
                if len(roi) == 40:
                    print("Writing image %s" % img_name)
                    cv2.imwrite(img_name, roi)
                else:
                    print("Dwarf image, not writing...")

            if not args['learning']:
                if args['poi']:
                    preds = 1
                    if cx != 0 and cy != 0:
                        if cv2.contourArea(con) > 25:
                            poi.draw_poi(image, cx-x_offset,
                                         cy-y_offset,config,preds)


def scale_image(image, args):
    """
    Scales image up or down based on '-s' args

    Author: robertdcurrier@gmail.com
    Created:    2018-11-06
    Modified:   2018-11-06
    """
    image = cv2.resize(image, (0, 0), fx=float(args["zoom"]),
                       fy=float(args["zoom"]))
    return image


def caption_image(image, config, scale, taxa, cell_count):
    """
    Puts caption on images. Need to add date of most recent
    video processing along with date/time of capture
    """
    the_date = time.strftime('%c')
    # Title
    the_text = config['captions']['title'] + config['captions']['version']
    x_pos = config['captions']['caption_x']
    y_pos = config['captions']['caption_y']
    cap_font_size = config['captions']['cap_font_size']
    cv2.putText(image, the_text, (x_pos, y_pos),
                cv2.FONT_HERSHEY_PLAIN, cap_font_size,
                (255, 255, 255))
    y_pos = y_pos + 20
    # Date/Time
    cv2.putText(image, the_date, (x_pos, y_pos),
                cv2.FONT_HERSHEY_PLAIN, cap_font_size, (255, 255, 255))

    # Model
    y_pos = y_pos + 20
    the_text = "Taxa: %s" % taxa
    cv2.putText(image, the_text, (x_pos, y_pos),
                cv2.FONT_HERSHEY_PLAIN, cap_font_size, (255, 255, 255))

    # Cell count
    y_pos = y_pos + 20
    the_text = "Cells: %s" % cell_count
    cv2.putText(image, the_text, (x_pos, y_pos),
                cv2.FONT_HERSHEY_PLAIN, cap_font_size, (255, 255, 255))

    # Max cells
    y_pos = y_pos + 20
    the_text = "Max Cells: %s" % MAX_CELLS
    cv2.putText(image, the_text, (x_pos, y_pos),
                cv2.FONT_HERSHEY_PLAIN, cap_font_size, (255, 255, 255))

    # CPL
    y_pos = y_pos + 20
    cpl_count = calc_cellcount(config, scale)
    the_text = "Estimated c/L: %d" % (cpl_count)
    cv2.putText(image, the_text, (x_pos, y_pos),
                cv2.FONT_HERSHEY_PLAIN, cap_font_size, (255, 255, 255))
    return image


def show_image(image):
    """
    D'oh. Shows the image w/delay

    Author: robertdcurrier@gmail.com
    Created:    2018-11-06
    Modified:   2018-11-06
    """
    print("show_image()")
    args = get_args()
    cv2.imshow('PhytoTracker', image)
    while True:
        keycode = int(cv2.waitKey(1)) & 255
        if keycode == 27:
            sys.exit()

def calc_cellcount(config, scale):
    """
    Calculates eCPL based on interpolated scale

    Author: robertdcurrier@gmail.com
    Created:    2018-11-06
    Modified:   2018-11-06
    """
    global MAX_CELLS
    taxa = config['system']['default_taxa']
    max_cells_cutoff = config['taxa'][taxa]['max_cells_cutoff']
    if MAX_CELLS < 1:
        return 0
    if MAX_CELLS > 0:
        if MAX_CELLS > max_cells_cutoff:
            MAX_CELLS = max_cells_cutoff
        interp_cells = scale['scale'][MAX_CELLS - 1]
        return interp_cells

def init_app():
    """
    Kick it!
    """
    config = get_config()
    args = get_args()

    if (args["input"]) is None:
        show_info()

    if args["taxa"]:
        validate_taxa(args)
        taxa = args["taxa"]
    else:
        taxa = config['system']['default_taxa']

    # Set backend
    keras.backend.set_image_data_format("channels_first")
    model = load_model()
    process_image(args["input"], model, taxa)


if __name__ == '__main__':
    init_app()
