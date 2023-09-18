import os
import shutil
import sys
import tensorflow as tf

dir = tf.sysconfig.get_link_flags()[0][2:]
name = tf.sysconfig.get_link_flags()[1][3:]

shutil.copy(os.path.realpath(dir) + "/" + name, sys.argv[1])
