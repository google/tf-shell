import os
import shutil
import sys
import tensorflow as tf

include_dir = tf.sysconfig.get_compile_flags()[0][2:]

shutil.copytree(os.path.realpath(include_dir), sys.argv[1] + "/include")
