
import numpy as np
import tensorflow as tf

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def inference(image):
  '''This function defines CNN model and builds graph for inference
  Args:
  image: Input image node with image of shape [?, 28, 28, 1]
  '''
  
  # Layer 1: 5x5 convolution, 32 feature maps, relu and 2x2 max-pooling
  w1 = weight_variable([5, 5, 1, 32])
  b1 = bias_variable([1, 32])
  
  conv1 = tf.nn.relu(tf.nn.conv2d(image, w1, strides = [1, 1, 1, 1], padding = 'SAME') + b1)
  pool1 = tf.nn.max_pool(conv1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
  
  # Layer 2: 5x5 convolution, 64 feature maps, relu and 2x2 max-pooling
  w2 = weight_variable([5, 5, 32, 64])
  b2 = bias_variable([1, 64])
  
  conv2 = tf.nn.relu(tf.nn.conv2d(pool1, w2, strides = [1, 1, 1, 1], padding = 'SAME') + b2)
  pool2 = tf.nn.max_pool(conv2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
  
  # Layer 3: Dense layer with 1024 neurons
  w3 = weight_variable([7 * 7 * 64, 1024])
  b3 = bias_variable([1, 1024])
  
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
  fc3 = tf.nn.relu(tf.matmul(pool2_flat, w3) + b3)
  
  # Layer 4: Final layer with 10 class
  w4 = weight_variable([1024, 10])
  b4 = bias_variable([1, 10])
  
  fc4 = tf.matmul(fc3, w4) + b4
  
  return fc4

