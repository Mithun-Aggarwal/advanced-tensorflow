#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 19:24:14 2017

@author: MIT
"""

import os
from config import config
import tensorflow as tf
import numpy as np
from model import Model

#import ops

from tfrecords_reader import TFRecordsReader

tf.app.flags.DEFINE_integer('checkpoint_period', 10000,
                            "Number of batches in between checkpoints")
tf.app.flags.DEFINE_float('epsilon', 1e-8,
                          "Fuzz term to avoid numerical instability")
tf.app.flags.DEFINE_string('run', 'demo',
                            "Which operation to run. [demo|train]")
tf.app.flags.DEFINE_float('gene_l1_factor', 0.8,
                          "Multiplier for generator L1 loss term")
tf.app.flags.DEFINE_float('learning_beta1', 0.5,
                          "Beta1 parameter used for AdamOptimizer")
tf.app.flags.DEFINE_float('learning_rate_start', 0.00020,
                          "Starting learning rate used for AdamOptimizer")
tf.app.flags.DEFINE_integer('learning_rate_half_life', 5000,
                            "Number of batches until learning rate is halved")
tf.app.flags.DEFINE_bool('log_device_placement', False,
                         "Log the device where variables are placed.")
tf.app.flags.DEFINE_integer('sample_size', 64,
                            "Image sample size in pixels. Range [64,128]")
tf.app.flags.DEFINE_integer('summary_period', 4000, # 10,000
                            "Number of batches between summary data dumps")
tf.app.flags.DEFINE_integer('random_seed', 0,
                            "Seed used to initialize rng.")
tf.app.flags.DEFINE_integer('test_vectors', 16,
                            """Number of features to use for testing""")
tf.app.flags.DEFINE_string('train_dir', 'train',
                           "Output folder where training logs are dumped.")
tf.app.flags.DEFINE_integer('train_time', 30,
                            "Time in minutes to train the model")
tf.app.flags.DEFINE_integer('max_batch', 20000,"Maximum number of batch iterations")
tf.app.flags.DEFINE_string('dataset', 'LF', 'name of data set')
tf.app.flags.DEFINE_integer('input_height', config.ORIGNAL.IMG_H_PATCH, 'input image height')
tf.app.flags.DEFINE_integer('input_width', config.ORIGNAL.IMG_W_PATCH, 'input image width')
tf.app.flags.DEFINE_integer('input_channels', config.ORIGNAL.IMG_C_PATCH, 'image channels')
tf.app.flags.DEFINE_integer('input_image_height', config.DCGAN.input_image_height, 'input image height')
tf.app.flags.DEFINE_integer('input_image_width', config.DCGAN.input_image_width, 'input image width')
tf.app.flags.DEFINE_integer('input_image_chanel', config.DCGAN.input_image_channel, 'input image chanel')
tf.app.flags.DEFINE_integer('output_height', config.DCGAN.output_image_height, 'output image height')
tf.app.flags.DEFINE_integer('output_width', config.DCGAN.output_image_width, 'output image width')
tf.app.flags.DEFINE_integer('z_dim', config.GENERATOR.D_PATCH, 'generator input dim')
tf.app.flags.DEFINE_integer('num_filters', config.SETTINGS.filters, 'number of filters')
tf.app.flags.DEFINE_boolean('crop', config.SETTINGS.crop, 'crop image or not')
tf.app.flags.DEFINE_integer('batch_size', config.TRAIN.batch_size, 'batch size')
tf.app.flags.DEFINE_float('learning_rate', config.TRAIN.lr, 'learning rate')
tf.app.flags.DEFINE_float('beta1', config.TRAIN.beta1, 'momentum term of Adam')

FLAGS = tf.app.flags.FLAGS

def inference(images, z):
    
    generated_images = generator(z)
    D_logits_real = discriminator(images)

    D_logits_fake = discriminator(generated_images, reuse=True)

    return D_logits_real, D_logits_fake, generated_images


def _generator_model(sess, features, labels, channels):
    # Upside-down all-convolutional resnet
    mapsize = 3
    res_units  = [256, 128, 96]
    old_vars = tf.global_variables()
    # See Arxiv 1603.05027
    model = Model('GEN', features)
    for ru in range(len(res_units)-1):
        nunits  = res_units[ru]
        for j in range(2):
            model.add_residual_block(nunits, mapsize=mapsize)
        # Spatial upscale (see http://distill.pub/2016/deconv-checkerboard/)
        # and transposed convolution
        model.add_upscale()
        model.add_batch_norm()
        model.add_relu()
        model.add_conv2d_transpose(nunits, mapsize=mapsize, stride=1, stddev_factor=1.)
    # Finalization a la "all convolutional net"
    nunits = res_units[-1]
    model.add_conv2d(nunits, mapsize=mapsize, stride=1, stddev_factor=2.)
    # Worse: model.add_batch_norm()
    model.add_relu()
    model.add_conv2d(nunits, mapsize=1, stride=1, stddev_factor=2.)
    # Worse: model.add_batch_norm()
    model.add_relu()
    # Last layer is sigmoid with no batch normalization
    model.add_conv2d(channels, mapsize=1, stride=1, stddev_factor=1.)
    model.add_sigmoid()
    new_vars  = tf.global_variables()
    gene_vars = list(set(new_vars) - set(old_vars))
    print ("GENERATOR READY")
    return model.get_output(), gene_vars
    

def _discriminator_model(sess, features, disc_input):
    # Fully convolutional model
    mapsize = 3
    layers  = [64, 128, 256, 512]
    old_vars = tf.global_variables()
    model = Model('DIS', 2*disc_input - 1)
    for layer in range(len(layers)):
        nunits = layers[layer]
        stddev_factor = 2.0
        model.add_conv2d(nunits, mapsize=mapsize, stride=2, stddev_factor=stddev_factor)
        model.add_batch_norm()
        model.add_relu()
    # Finalization a la "all convolutional net"
    model.add_conv2d(nunits, mapsize=mapsize, stride=1, stddev_factor=stddev_factor)
    model.add_batch_norm()
    model.add_relu()
    model.add_conv2d(nunits, mapsize=1, stride=1, stddev_factor=stddev_factor)
    model.add_batch_norm()
    model.add_relu()
    # Linearly map to real/fake and return average score
    # (softmax will be applied later)
    model.add_conv2d(1, mapsize=1, stride=1, stddev_factor=stddev_factor)
    model.add_mean()
    new_vars  = tf.global_variables()
    disc_vars = list(set(new_vars) - set(old_vars))
    print ("DISCRIMINATOR READY")
    return model.get_output(), disc_vars
    

def inputs(batch_size=64):
    crop = FLAGS.crop
    image_height, image_width = FLAGS.input_height, FLAGS.input_width    
    crop_height, crop_width = FLAGS.input_image_height, FLAGS.input_image_width
    resize_height, resize_width = FLAGS.output_height, FLAGS.output_width

    if FLAGS.dataset == "LF":
        reader = TFRecordsReader(
            image_height=image_height,
            image_width=image_width,
            image_channels=3,
            image_format="png",
            directory=config.PRE_TRAIN.output_directory,
            filename_pattern="LF-*",
            crop=crop,
            crop_height=crop_height,
            crop_width=crop_width,
            resize=True,
            resize_height=resize_height,
            resize_width=resize_width,
            num_examples_per_epoch=64)

        images, _ = reader.inputs(batch_size=64)
        float_images = tf.cast(images, tf.float32)
        float_images = float_images / 127.5 - 1.0