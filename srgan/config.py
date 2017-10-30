# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 22:35:32 2017

@author: ksxl806
"""

from easydict import EasyDict as edict

config = edict()
config.PRE_TRAIN = edict()
config.TRAIN = edict()
config.DCGAN = edict()
config.ORIGNAL = edict()
config.GENERATOR = edict()
config.SETTINGS = edict()

# Data Config 375, 541
config.ORIGNAL.IMG_H_PATCH = 375
config.ORIGNAL.IMG_W_PATCH = 541
config.ORIGNAL.IMG_C_PATCH = 3
#config.ORIGNAL.IMG_PATCH = config.IMG_H_PATCH * config.IMG_W_PATCH

#DCGAN-INPORT
config.DCGAN.input_image_height = 300
config.DCGAN.input_image_width = 300
config.DCGAN.input_image_channel = 3

config.DCGAN.output_image_height = 160
config.DCGAN.output_image_width = 160
#config.DCGAN.input_image_channel = 3

#DGGAN-Generator
config.GENERATOR.D_H_PATCH = 10  
config.GENERATOR.D_W_PATCH = 10
config.GENERATOR.D_PATCH = 100
config.GENERATOR.D_C_PATCH = 1

#DGGAN-Generator

#DCGAN-SETTINGS
config.SETTINGS.filters = 64
config.SETTINGS.crop = 'True'


## Training (LF)
config.TRAIN.steps = 1000
config.TRAIN.batch_size = 64
config.TRAIN.lr_decay = 0.5
config.TRAIN.decay_every = 1000
config.TRAIN.lr = 0.0002
config.TRAIN.beta1 = 0.5

## PRE_TRAINING
config.PRE_TRAIN.directory = '/Users/MIT/Downloads/LF_DCGAN/RAW_IMAGE'
config.PRE_TRAIN.output_directory = '/Users/MIT/Downloads/LF_SRGAN/LF_TFrecord'
config.PRE_TRAIN.shards = 64
config.PRE_TRAIN.threads = 8
config.PRE_TRAIN.import_colorspace = 'RGB'
config.PRE_TRAIN.import_channels = 3
config.PRE_TRAIN.import_image_format = 'PNG'


## Train Folder
#config.TRAIN.img_path = 'C:/Users/ksxl806/Documents/PythonScripts/TenserflowGit/LF_superresolution_python3_code/LF_superresolution_python3_code/Data/Data/'
#config.TRAIN.hr_img_path = 'C:/Users/ksxl806/Documents/PythonScripts/TenserflowGit/LF_superresolution_python3_code/LF_superresolution_python3_code/Data/Train/'

## Validation Folder
#config.VALID = edict()
#config.VALID.hr_img_path = 'C:/Users/ksxl806/Documents/PythonScripts/TenserflowGit/LF_superresolution_python3_code/LF_superresolution_python3_code/Data/Valid/'

## Test Folder
##config.TEST = edict()
#config.TEST.hr_img_path = ''


config.G_H_PATCH = 100
config.G_W_PATCH = 300
config.G_PATCH = config.G_H_PATCH * config.G_W_PATCH
config.G_C_PATCH = 3


config.T_CHANEL = [1]
#config.SCALE = 2
config.T_PATCH = 50
config.CAPACITY = 100
config.min_after_dequeue = 10
config.DIM = [14,14]

# Save Directories
#config.SAVE_RESULT = "C:/Users/ksxl806/Documents/PythonScripts/GAN_TF_PYT/LF_GAN/samples/LapSRN"
#config.SAVE_MODEL = "C:/Users/ksxl806/Documents/PythonScripts/GAN_TF_PYT/LF_GAN/checkpoint_l1_loss"
config.LOG_LOCATION = "/Users/MIT/Downloads/LF_SRGAN/TLog"
config.CHECKPOINT_LOCATION = "/Users/MIT/Downloads/LF_SRGAN/srgan/checkpoint"
#config.SAVE_IMAGE = "C:/Users/ksxl806/Documents/PythonScripts/GAN_TF_PYT/LF_GAN/IMAGE"
