#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 10:52:31 2017

@author: MIT
"""

import datetime
import os

import numpy as np
import tensorflow as tf

import lfsrgan
from config import config
from utils import *


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('train_steps', config.TRAIN.steps, 'number of train steps')

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      tf.summary.histogram('histogram', var)
      
def train():
    # placeholder for z
    z = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.z_dim], name='z')

    # get images
    images = lfsrgan.inputs(batch_size=FLAGS.batch_size)

    # logits
    with tf.name_scope("logits"):
        D_logits_real, D_logits_fake, generated_images = dcgan.inference(images, z)
        with tf.name_scope("D_logits_real"):
            variable_summaries(D_logits_fake)
        with tf.name_scope("D_logits_real"):
            variable_summaries(D_logits_fake)
        with tf.name_scope("Generated_image"):
            tf.summary.image('Generated Image',generated_images,3) 
            
    #images to tensorboard          
#==============================================================================
#     # loss
#     with tf.name_scope("loss"):
#         d_loss, g_loss = dcgan.loss(D_logits_real, D_logits_fake)
#         with tf.name_scope("d_loss"):
#             variable_summaries(d_loss)
#         with tf.name_scope("g_loss"):
#             variable_summaries(g_loss)
#             
#     # train the model
#     with tf.name_scope("train_ops"):
#         train_d_op, train_g_op = dcgan.train(d_loss, g_loss)
#             
#     init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
#     #Runtime options
#     summary_op = tf.summary.merge_all()
#     run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
#     run_metadata = tf.RunMetadata()
#     
#     #GPU_&_Computation
#     gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
#     coord = tf.train.Coordinator()
#     with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
#         summary_writer = tf.summary.FileWriter(config.LOG_LOCATION+'/LF_DCGAN',sess.graph)
#         sess.run(init)
#         threads = tf.train.start_queue_runners(sess=sess,coord=coord)
# 
#         saver = tf.train.Saver()
# 
#         training_steps = FLAGS.train_steps
# 
#         for step in range(training_steps):
# 
#             random_z = np.random.uniform(
#                 -1, 1, size=(FLAGS.batch_size, FLAGS.z_dim)).astype(np.float32)
# 
#             
#             sess.run(train_d_op, feed_dict={z: random_z})
#             sess.run(train_g_op, feed_dict={z: random_z})
#             sess.run(train_g_op, feed_dict={z: random_z})
#             
#             with tf.name_scope("Loss"):
#                 discrimnator_loss, generator_loss = sess.run(
#                     [d_loss, g_loss], feed_dict={z: random_z})
#                 with tf.name_scope("discrimnator_loss"):
#                     variable_summaries(discrimnator_loss)
#                 with tf.name_scope("generator_loss"):
#                     variable_summaries(generator_loss)
#                 
#             time_str = datetime.datetime.now().isoformat()
#             print("{}: step {}, d_loss {:g}, g_loss {:g}".format(
#                 time_str, step, discrimnator_loss, generator_loss))
#             
#             summary_writer.add_run_metadata(run_metadata, 'step%03d' % step)
#             
#             test_images = sess.run(generated_images, feed_dict={z: random_z})
# 
#             image_path = os.path.join(FLAGS.checkpoint_dir,
#                                       "sampled_images_%d.png" % step)
#             
#             summary = sess.run(summary_op, feed_dict={z: random_z},
#                                options=run_options,
#                                run_metadata=run_metadata)
#             summary_writer.add_summary(summary,step) 
# 
#             if step % 100 == 0:
#                 utils.grid_plot(test_images, [8, 8], image_path)
#                 saver.save(sess, os.path.join(FLAGS.checkpoint_dir, "model.ckp"))    
# 
#     
#     coord.request_stop()
#     coord.join(threads)
#     summary_writer.close()
#==============================================================================




def main(argv=None):
    
#==============================================================================
#     all_filenames = prepare_dirs(delete_train_dir=True)
#     print ("[%d] IMAGES LOADED." % (len(all_filenames)))
#==============================================================================

    train()


if __name__ == '__main__':
    tf.app.run()
