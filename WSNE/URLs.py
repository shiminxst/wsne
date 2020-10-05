# -*- coding: utf-8 -*-

# define the local paths
import tensorflow as tf


# Web service dataset
# tf.flags.DEFINE_string("serRootDir","C:/MountedDisk/git/workspace_python/ServiceNE/dataset","root dir")
tf.flags.DEFINE_string("serviceData","./datasets/API","Web API dataset")

# # Cora dataset
# tf.flags.DEFINE_string("coraData","./datasets/cora","coraData dataset")

# Citeseer dataset
tf.flags.DEFINE_string("citeseerData","./datasets/citeseer","citeseer dataset")

# wiki dataset
tf.flags.DEFINE_string("wikiData","./datasets/wiki","wiki dataset")


FILES = tf.flags.FLAGS