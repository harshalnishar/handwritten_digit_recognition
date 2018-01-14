import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

import argparse
import os

import mnist_model

parser = argparse.ArgumentParser()

parser.add_argument('--image', type=str, default='test_image.jpg',
                    help='Path to the image you want to test')

FLAGS = parser.parse_args()

img = cv2.imread(FLAGS.image)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
(height, width) = img.shape
max_dim = max(height, width)

img = cv2.normalize(img, 0, 255, norm_type = cv2.NORM_MINMAX)

img_median = cv2.medianBlur(img, 7)
img_blurred1 = cv2.bilateralFilter(img_median, 9, 15, 30)

kernel_size = int(max_dim / (1.5 * 28))
kernel = np.ones((kernel_size, kernel_size), np.uint8)
img_erode1 = cv2.erode(img_blurred1, kernel)

if height != width:
  if height > width:
    pad_size = int((height - width) / 2)
    img_padded1 = cv2.copyMakeBorder(img_erode1, 0, 0, pad_size, pad_size, cv2.BORDER_REPLICATE)
  else:
    pad_size = int((width - height) / 2)
    img_padded1 = cv2.copyMakeBorder(img_erode1, pad_size, pad_size, 0, 0, cv2.BORDER_REPLICATE)
else:
  img_padded1 = img_erode1

img_downscale1 = cv2.resize(img_padded1, (28, 28), interpolation=cv2.INTER_CUBIC)

img_threshold1 = cv2.adaptiveThreshold(img_downscale1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 30)

img_prep1 = cv2.bitwise_not(img_threshold1)
img_prep1 = (img_prep1 / 255).astype('float32')

plt.gray()
plt.subplot(2, 2, 1), plt.imshow(img_blurred1)
plt.subplot(2, 2, 2), plt.imshow(img_erode1)
plt.subplot(2, 2, 3), plt.imshow(img_downscale1)
plt.subplot(2, 2, 4), plt.imshow(img_threshold1)
plt.show()

# Placeholder for output prediction
x = tf.placeholder(shape = [1, 28 , 28, 1], dtype = tf.float32)

logits = tf.nn.softmax(mnist_model.inference(x));

saver = tf.train.Saver()
sess = tf.Session();
saver.restore(sess, './mnist_train/model.ckpt')

img_final = img_prep1.reshape([1, 28, 28, 1])
prediction = (sess.run(logits, feed_dict = {x: img_final}))
number = np.argmax(prediction)
print('Your image contains: %d, with confidence: %f' % (number, prediction[0,number]))

