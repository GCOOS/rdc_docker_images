#!/usr/bin/env python3
# coding: utf-8
"""This is the NVIDIA GPU version.

Modified: 2020-10-02
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

# Helper libraries
import json
import time
import sys
import os
import logging
import ffmpeg
import argparse
import cv2 as cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

import poi_target as poi
# globals
MAX_CELLS = 0
THUMB_COUNT = 0

def string_to_tuple(the_str):
    """
    Converts string to tuple for use in OpenCV
    drawing functions. Sort of an ass-backswards methond, but...

    Author: robertdcurrier@gmail.com
    Created:    2018-11-07
    Modified:   2018-11-07
    """
    color = []
    tup = map(int, the_str.split(','))
    for val in tup:
        color.append(val)
    color = tuple(color)
    return color


def show_info():
    """Print info when called from command line w/no options.

    Author: robertdcurrier@gmail.com
    Created:    2018-11-06
    Modified:   2018-11-06
    """
    logging.warn("show_info(): No input file specified.")
    config = get_config()
    logging.warn(config['captions']['title'])
    logging.warn("Testing load_model()")
    load_model()
    logging.warn("Testing load_scale()")
    load_scale()
    # Goombye
    sys.exit()


def get_config():
    """Get config settings from config.json.

    Author: robertdcurrier@gmail.com
    Created:    2019-07-18
    Modified:   2019-07-18
    """
    c_file = open('configs/phyto3.cfg').read()
    config = json.loads(c_file)
    return config


def load_model():
    """Load TensorFlow model and cell weights from lib.

    Author: robertdcurrier@gmail.com
    Created:    2019-07-18
    Modified:   2018-07-18
    """
    config = get_config()
    args = get_args()

    if args["taxa"]:
        taxa = args["taxa"]
        model_file = config['taxa'][taxa]['model_file']
    else:
        taxa = config['system']['default_taxa']
        model_file = config['taxa'][taxa]['model_file']

    if(config['system']['debug']):
         logging.warn("Using %s as taxa and %s as model_file" % (taxa, model_file))

    try:
        json_file = open(model_file)
        if(config['system']['debug']):
            logging.warn("load_model(): Loaded %s model_file successfully" % taxa)
    except IOError:
        logging.warn("load_model(): Failed to open %s model_file" % model_file)
        sys.exit()

    model_json = json_file.read()
    json_file.close()
    model = tensorflow.keras.models.model_from_json(model_json)
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
        logging.warn("load_model(): Failed to open %s" % weights_file)
        sys.exit()
    return model


def load_scale():
    """Load scale file from lib.

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
        logging.warn("load_scale(): Failed to open %s" % scale_file)
        sys.exit()

    scale = json.loads(scale_file.read())
    scale_file.close()
    if(config['system']['debug']):
        logging.warn("Loaded %s scale successfully" % taxa)
    return scale


def get_args():
    """Get command line args.

    Author: robertdcurrier@gmail.com
    Created:    2018-11-06
    Modified:   2020-10-02
    """
    arg_p = argparse.ArgumentParser()
    arg_p.add_argument("-i", "--input",
                       help="Path to the input video file")
    arg_p.add_argument("-o", "--output",
                       help="Path to the output video file")
    arg_p.add_argument("-z", "--zoom", help="zoom factor")
    arg_p.add_argument("-l", "--learning", help="generate training sets",
                       action="store_true")
    arg_p.add_argument("-t", "--taxa", help="choose taxa")
    arg_p.add_argument("-c", "--contours", help="show contours",
                       action="store_true")
    args = vars(arg_p.parse_args())
    return args


def validate_taxa(args):
    """Confirm taxa is in config file."""
    config = get_config()
    taxa = args["taxa"]
    if config['system']['debug']:
        logging.warn("validate_taxa(): Validating %s" % taxa)
    if taxa in config['taxa'].keys():
        logging.warn("validate_taxa(): Found %s taxa settings" % taxa)
    else:
        logging.warn("validate_taxa(): %s not found!" % taxa)
        sys.exit()


def process_video(file_name, model, taxa):
    """Process the sucker.

    Read video input file, gets cons list,
    draws target indicator and updates settings. Skips first n frames
    and last n frames to avoid shaking. N defined in config file

    Author: robertdcurrier@gmail.com
    Created:    2018-11-06
    Modified:   2018-11-06
    """
    # local variables
    frame_count = 0
    # get settings
    args = get_args()

    config = get_config()
    scale = load_scale()
    logging.warn("process_video(): Using taxa %s" % taxa)

    video_file = cv2.VideoCapture(file_name)
    size = (int(video_file.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(video_file.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    max_frames = (int(video_file.get(cv2.CAP_PROP_FRAME_COUNT)))
    skip_frames = config['taxa'][taxa]['skip_frames']
    thresh_min = config['taxa'][taxa]['thresh_min']
    thresh_max = config['taxa'][taxa]['thresh_max']
    MOG_thresh = config['taxa'][taxa]['MOG_thresh']
    MOG_history = config['taxa'][taxa]['MOG_history']
    learning_rate = config['taxa'][taxa]['learning_rate']
    con_min = config['taxa'][taxa]['con_min']
    con_max = config['taxa'][taxa]['con_max']
    max_cons = config['taxa'][taxa]['max_cons']
    if config['system']['debug']:
        logging.warn("process_video(): max_frames %d" % max_frames)
        logging.warn("process_video(): loading %s" % file_name)
        logging.warn("process_video(): Skipping %d frames" % skip_frames)
        logging.warn("process_video(): thresh_min %d" % thresh_min)
        logging.warn("process_video(): thresh_max %d" % thresh_max)
        logging.warn("process_video(): MOG_history %d" % MOG_history)
        logging.warn("process_video(): MOG_threshold %d" % MOG_thresh)
        logging.warn("process_video(): con_min %d" % con_min)
        logging.warn("process_video(): con_max %d" % con_max)
        logging.warn("process_video(): max_cons %d" % con_max)
        logging.warn("process_video(): learning_rate %0.6f" % learning_rate)
    background_sub = (cv2.createBackgroundSubtractorMOG2(
        detectShadows=False, history=MOG_history,
        varThreshold=MOG_thresh))
    if args["output"]:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args["output"], fourcc, 24, size)

    # spin through the video
    while frame_count < (max_frames - skip_frames):
        _, frame = video_file.read()
        # Equalize histogram and get contours
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # blurred = cv2.GaussianBlur(gray, (11, 11), 0)
        fgmask = background_sub.apply(gray, learningRate=learning_rate)
        threshold = cv2.threshold(fgmask, thresh_min, thresh_max,
                                  cv2.THRESH_BINARY)[1]
        contours, _ = (cv2.findContours(threshold.copy(), cv2.RETR_TREE,
                                        cv2.CHAIN_APPROX_NONE))

        # get matched contours
        good_cons = gen_contours(contours, config, taxa)
        # Note: MUST do target before zoom or we get all borked
        cell_count = target_cells(frame, good_cons, taxa, config, model)
        if not args['learning']:
            frame, cpl_count = caption_frame(frame, config, scale, taxa, cell_count)

        if args["contours"]:
            con_color = config['taxa'][taxa]['con_color']
            con_color = string_to_tuple(con_color)
            cv2.drawContours(frame, good_cons, -1, con_color, 1)
        if args["output"]:
            out.write(frame)
        frame_count += 1

    return cpl_count


def gen_contours(contours, config, taxa):
    """Generate contour list for each frame based on config settings.

    Author: robertdcurrier@gmail.com
    Created:    2018-11-06
    Modified:   2018-11-07
    """
    good_cons = []
    con_min = config['taxa'][taxa]['con_min']
    con_max = config['taxa'][taxa]['con_max']
    max_cons = config['taxa'][taxa]['max_cons']
    for con in contours:
        area = cv2.contourArea(con)
        if area > con_min and area < con_max:
            if len(con) < max_cons:
                good_cons.append(con)
    return good_cons


def classify_cell(model, cell):
    """Classify cells.

    Author: robertdcurrier@gmail.com
    Created:    2018-11-06
    Modified:   2019-04-05

    We changed this up a bit... we are now passing buffers,
    not writing to file for cell images. We reshape to
    64, 64 to match the model. If mode.predict returns none for
    some reason we return a 2 so we can ignore or flag. Brevis
    returns a 0 and Not Brevis returns a 1.
    """
    if cell.shape == (40, 40, 3):
        cell = cv2.resize(cell, (40, 40))
        x = img_to_array(cell)
        x = x.reshape((1,) + x.shape)
        x = preprocess_input(x)
        preds = (model.predict_classes(x, verbose=0))
        if preds[0][0] == 0:
            return False
        if preds[0][0] == 1:
            return True


def target_cells(frame, good_cons, taxa, config, model):
    """Draw 'Person of Interest' target around detected cells.

    if TensorFlow model returns True. If False then draw
    a red rectangle if '-a' option set.

    Author: robertdcurrier@gmail.com
    Created:    2018-11-06
    Modified:   2020-09-24
    """
    global MAX_CELLS, THUMB_COUNT
    x_offset = config['taxa'][taxa]['x_offset']
    y_offset = config['taxa'][taxa]['y_offset']
    roi_offset = config['taxa'][taxa]['roi_offset']
    roi_min = config['taxa'][taxa]['roi_min']
    roi_max = config['taxa'][taxa]['roi_max']
    max_thumbs = config['taxa'][taxa]['max_thumbs']
    preds = None
    roi_match = False
    cell_count = 0
    args = get_args()
    config = get_config()

    for con in good_cons:
        (x_pos, y_pos, w, h) = cv2.boundingRect(con)
        # logging.warn(cv2.contourArea(con))
        if((w > roi_min) and (w < roi_max) and (h > roi_min) and (h < roi_max)):
            roi_match = True
            cx = 0
            cy = 0
            M = cv2.moments(con)
        else:
            roi_match = False
            continue

        if int(M['m10']) > 0 and int(M['m01']) > 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            roi = (frame[cy-roi_offset:cy+roi_offset,
                         cx-roi_offset:cx+roi_offset])
            if roi.size != 0:
                roi = cv2.resize(roi, (40, 40))
            epoch = int(time.time()*1000)
            if args['learning']:
                # GET THIS IN CONFIG.JSON YO
                if THUMB_COUNT > max_thumbs:
                    logging.warn('Max Thumbs exceeded.')
                    sys.exit()
                img_name = ('./tmp/%d.jpg' % epoch)
                if len(roi) == 40:
                    if(config['system']['debug']):
                        logging.warn("Writing image %s" % img_name)
                    cv2.imwrite(img_name, roi)
                    THUMB_COUNT+=1
                else:
                    if(config['system']['debug']):
                        logging.warn("Dwarf image, not writing...")
            if roi_match:
                preds = classify_cell(model, roi)
                if not args['learning']:
                    poi.draw_poi(frame, cx-x_offset,
                                 cy-y_offset, config, taxa, preds)
                if preds:
                    cell_count += 1
                    if cell_count > MAX_CELLS:
                        MAX_CELLS = cell_count
    return cell_count


def scale_frame(frame, args):
    """Scale frame up or down based on '-s' args.

    Author: robertdcurrier@gmail.com
    Created:    2018-11-06
    Modified:   2018-11-06
    """
    frame = cv2.resize(frame, (0, 0), fx=float(args["zoom"]),
                       fy=float(args["zoom"]))
    return frame


def caption_frame(frame, config, scale, taxa, cell_count):
    """Put caption on frames. Need to add date of most recent.

    video processing date/time and date/time of capture
    """
    the_date = time.strftime('%c')
    # Title
    version_text = "Version: "
    the_text = version_text + config['captions']['title'] + config['captions']['version']
    x_pos = config['captions']['caption_x']
    y_pos = config['captions']['caption_y']
    cap_font_size = config['captions']['cap_font_size']
    cv2.putText(frame, the_text, (x_pos, y_pos),
                cv2.FONT_HERSHEY_PLAIN, cap_font_size,
                (255, 0, 0))
    y_pos = y_pos + 20
    # Date/Time
    the_text = "Processed: %s" % the_date
    cv2.putText(frame, the_text, (x_pos, y_pos),
                cv2.FONT_HERSHEY_PLAIN, cap_font_size, (255, 0, 0))

    # Model
    y_pos = y_pos + 20
    the_text = "Taxa: %s" % taxa
    cv2.putText(frame, the_text, (x_pos, y_pos),
                cv2.FONT_HERSHEY_PLAIN, cap_font_size, (255, 0, 0))
    # Scale
    y_pos = y_pos + 20
    the_text = "Scale: %s" % scale["title"]
    cv2.putText(frame, the_text, (x_pos, y_pos),
                cv2.FONT_HERSHEY_PLAIN, cap_font_size, (255, 0, 0))

    # Cell count
    y_pos = y_pos + 20
    the_text = "Cells: %s" % cell_count
    cv2.putText(frame, the_text, (x_pos, y_pos),
                cv2.FONT_HERSHEY_PLAIN, cap_font_size, (255, 0, 0))

    # Max cells
    y_pos = y_pos + 20
    the_text = "Max Cells: %s" % MAX_CELLS
    cv2.putText(frame, the_text, (x_pos, y_pos),
                cv2.FONT_HERSHEY_PLAIN, cap_font_size, (255, 0, 0))

    # CPL
    y_pos = y_pos + 20
    cpl_count = calc_cellcount(config, taxa, scale)
    the_text = "Estimated c/L: %d" % (cpl_count)
    cv2.putText(frame, the_text, (x_pos, y_pos),
                cv2.FONT_HERSHEY_PLAIN, cap_font_size, (255, 0, 0))
    return frame, cpl_count


def calc_cellcount(config, taxa, scale):
    """Calculate eCPL based on interpolated scale.

    Author: robertdcurrier@gmail.com
    Created:    2018-11-06
    Modified:   2018-11-06
    """
    global MAX_CELLS
    max_cells_cutoff = config['taxa'][taxa]['max_cells_cutoff']
    if MAX_CELLS < 1:
        return 0
    if MAX_CELLS > 0:
        if MAX_CELLS > max_cells_cutoff:
            MAX_CELLS = max_cells_cutoff
        interp_cells = scale['scale'][MAX_CELLS - 1]
        return interp_cells


def init_app():
    """Kick it."""
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
    tensorflow.keras.backend.set_image_data_format("channels_last")
    model = load_model()
    cpl_count = process_video(args["input"], model, taxa)
    logging.warn("{'cpl' : %d}" % cpl_count)
    # Wouldn't run at the end of process_video as file hadn't closed properly
    ffmpeg_it(args["output"])


def ffmpeg_it(video_file):
    """OpenCV from pip can't do h264 so we have to here."""
    logging.warn('ffmpeg_it(%s)' % video_file)
    basename=(os.path.basename(video_file))
    ofile = '%s/%s_h264.mp4' % (os.path.dirname(video_file),
                                os.path.splitext(basename)[0])
    stream = ffmpeg.input(video_file)
    stream = stream.output(ofile)
    ffmpeg.run(stream)

if __name__ == '__main__':
    init_app()
