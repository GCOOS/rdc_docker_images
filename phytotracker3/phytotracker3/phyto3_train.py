#!/usr/bin/env python3
# coding: utf-8
"""NVIDA GPU version."""
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals
import sys
import json
import numpy
import argparse
import tensorflow as tf


def new_model():
    """Bigger model for testing."""
    # Create the model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), input_shape=(40, 40, 3),
                                     padding='same'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Conv2D(32, (3, 3),
                                     padding='same'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3),
                                     padding='same'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Conv2D(64, (3, 3),
                                     padding='same'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(128, (3, 3),
                                     padding='same'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Conv2D(128, (3, 3),
                                     padding='same'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(1024))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(512))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(1))
    model.add(tf.keras.layers.Activation('sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='rmsprop',
                  metrics=['accuracy'])

    return model


def phyto3_model():
    """Build Keras model for phytoplankton tracking."""
    config = get_config()
    taxa = config["system"]["default_taxa"]
    learning_rate = config["keras"][taxa]["learning_rate"]
    seed = 7
    numpy.random.seed(seed)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3, 3), input_shape=(40, 40, 3),
              data_format="channels_last"))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Conv2D(40, (3, 3), input_shape=(40, 40, 3),
              data_format="channels_last"))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Conv2D(128, (3, 3), input_shape=(40, 40, 3),
              data_format="channels_last"))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Flatten())  # convert 3D maps to 1D vectors
    model.add(tf.keras.layers.Dense(128))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(1))
    model.add(tf.keras.layers.Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='rmsprop',
                  metrics=['accuracy'], learning_rate = learning_rate)
    return model


def get_config():
    """D'oh."""
    c_file = open('configs/phyto3.cfg').read()
    config = json.loads(c_file)
    return config


def prepare_data(taxa):
    """Prepare and augment images."""
    config = get_config()
    imDataGen = tf.keras.preprocessing.image.ImageDataGenerator
    train_dir = config['taxa'][taxa]['train_dir']
    validation_dir = config['taxa'][taxa]['validation_dir']

    # this is the augmentation configuration we will use for TRAINING
    train_datagen = imDataGen(
        rotation_range=60,
        rescale=1./255,
        shear_range=0.5,
        zoom_range=0.5,
        horizontal_flip=True,
        fill_mode='nearest')

    # this is the augmentation configuration we will use for TESTING
    test_datagen = imDataGen(rescale=1. / 255)

    # this is a generator that will read pictures found in
    # subfolders of 'data/train', and indefinitely generate
    # batches of augmented image data
    train_generator = train_datagen.flow_from_directory(
        train_dir,  # this is the target directory
        # all images will be resized to 40x40
        target_size=(40, 40),
        batch_size=16,
        class_mode='binary')

    # this is a similar generator, for validation data
    validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(40, 40),
        batch_size=16,
        class_mode='binary')

    return train_generator, validation_generator


def train_model(taxa, model, train_generator, validation_generator):
    """Train, train, training we shall go."""
    config = get_config()
    steps_per_epoch = config['keras']['steps_per_epoch']
    epochs = config['keras']['epochs']
    validation_steps = config['keras']['validation_steps']
    model_file = config['taxa'][taxa]['model_file']
    weights_file = config['taxa'][taxa]['weights_file']
    print("Using %s and %s" % (model_file, weights_file))

    model.fit(
        train_generator,
        validation_data=validation_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_steps=validation_steps)

    # serialize model to JSON
    model_json = model.to_json()
    with open(model_file, "w") as json_file:
        json_file.write(model_json)
    model.save_weights(weights_file)
    print("Saved model to disk")


def validate_taxa(args):
    """Confirm taxa is in config file."""
    config = get_config()
    taxa = args["taxa"]
    if config['system']['debug']:
        print("validate_taxa(): Validating %s" % taxa)
    if taxa in config['taxa'].keys():
        print("validate_taxa(): Found %s taxa settings" % taxa)
    else:
        print("validate_taxa(): %s not found!" % taxa)
        sys.exit()


def get_args():
    """Get command line args.

    Author: robertdcurrier@gmail.com
    Created:    2018-11-06
    Modified:   2020-09-18
    """
    arg_p = argparse.ArgumentParser()
    arg_p.add_argument("-t", "--taxa", help="choose taxa", required="True")
    arg_p.add_argument("-e", "--epochs", help="number of epochs",
                       action="store_true")
    args = vars(arg_p.parse_args())
    return args


def init_app():
    """Kick it yo."""
    args = get_args()
    validate_taxa(args)
    taxa = args['taxa']

    model = new_model()
    tf.keras.backend.set_image_data_format('channels_last')
    train_generator, validation_generator = prepare_data(taxa)
    train_model(taxa, model, train_generator, validation_generator)

# test for lini9sfasdf YO~!

if __name__ == '__main__':
    init_app()
