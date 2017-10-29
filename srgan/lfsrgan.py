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

import ops

from tfrecords_reader import TFRecordsReader

FLAGS = tf.app.flags.FLAGS


def inference(images, z):
    generated_images = generator(z)
    D_logits_real = discriminator(images)

    D_logits_fake = discriminator(generated_images, reuse=True)

    return D_logits_real, D_logits_fake, generated_images