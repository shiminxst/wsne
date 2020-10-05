# -*- coding: utf-8 -*-

import tensorflow as tf

# define the paths of the datasets
# Webservice 
tf.flags.DEFINE_string("API", "../datasets/API", "")



tf.flags.DEFINE_float('emb_dropout_prob', 0.2, 'Dropout probability of embedding layer')
tf.flags.DEFINE_float('dropout_prob', 0.3, 'Dropout probability of output layer')

tf.flags.DEFINE_float('learning_rate', 2e-2, 'Initial learning rate.')
# tf.flags.DEFINE_float('weight_decay', 3e-3, 'Weight for L2 loss on embedding matri')
tf.flags.DEFINE_float('weight_decay', 3e-5, 'Weight for L2 loss on embedding matri')

FILES = tf.flags.FLAGS