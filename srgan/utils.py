import numpy as np
import scipy.misc
import tensorflow as tf
import numpy.random
from config import config

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('log_dir', config.LOG_LOCATION , 'log directory')
tf.app.flags.DEFINE_string('checkpoint_dir', config.CHECKPOINT_LOCATION , 'checkpoint directory')

def grid_plot(images, size, path):
    #images = (images + 1.0) / 2.0

    h, w = images.shape[1], images.shape[2]
    
    image_grid = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = int(idx / size[1])
        image_grid[int(j * h):int(j * h + h), int(i * w):int(i * w + w), :] = image

    scipy.misc.imsave(path, image_grid)

def prepare_dirs(delete_train_dir=False):
    # CREATE CHECKPOINT DIRECTORIES
    if not tf.gfile.Exists(FLAGS.checkpoint_dir):
        tf.gfile.MakeDirs(FLAGS.checkpoint_dir)
        print ("CREATE CHECKPOINT FOLDER[%s]" % (FLAGS.checkpoint_dir))
    else:
        print ("CHECKPOINT FOLDER[%s] ALREADY EXISTS" % (FLAGS.checkpoint_dir))
    # CLEANUP TRAIN DIR
    if delete_train_dir:
        if tf.gfile.Exists(FLAGS.train_dir):
            tf.gfile.DeleteRecursively(FLAGS.train_dir)
            print ("DELETE EVERY FILES IN TRAIN FOLDER[%s]" % (FLAGS.train_dir))
        tf.gfile.MakeDirs(FLAGS.train_dir)
        print ("CREATE TRAIN FOLDER[%s]" % (FLAGS.train_dir))
        
    # Return names of training files
    if not tf.gfile.Exists(FLAGS.dataset) or \
        not tf.gfile.IsDirectory(FLAGS.dataset):
        print ("DATASET FOLDER[%s] DOES NOT EXIST" % (FLAGS.dataset))
        return
    else:
        print ("DATASET FOLDER[%s] EXISTS" % (FLAGS.dataset))
    # LOAD FILES IN THE DATASET FOLDER
    filenames = tf.gfile.ListDirectory(FLAGS.dataset)
    filenames = sorted(filenames)
    random.shuffle(filenames)
    filenames = [os.path.join(FLAGS.dataset, f) \
                 for f in filenames if os.path.splitext(f)[1] == '.jpg']
    return filenames

def setup_tensorflow():
    # Create session
    config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=config)
    # Initialize rng with a deterministic seed
    with sess.graph.as_default():
        tf.set_random_seed(FLAGS.random_seed)
    random.seed(FLAGS.random_seed)
    np.random.seed(FLAGS.random_seed)
    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
    return sess, summary_writer