
import tensorflow as tf
import mnist_model
from tensorflow.examples.tutorials.mnist import input_data

def train(input):
  ''' Trains the CNN model defined in minst_model using gradient decent optimizer
  
  Args:
  '''
  
  # Place holders for input data and output predictions
  x = tf.placeholder(tf.float32, shape = [None, 784])
  y = tf.placeholder(tf.float32, shape = [None, 10])

  image = tf.reshape(x, [-1, 28, 28, 1])

  prediction = mnist_model.inference(image)
  cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = prediction))
  opt = tf.train.GradientDescentOptimizer(0.01)
  train_step = opt.minimize(cost)
  
  correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  saver = tf.train.Saver()
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(50000):        
      batch = input.train.next_batch(100)
      if i % 100 == 0:
          print('Step = %d, Train accuracy = %f' % (i, accuracy.eval(feed_dict = {x: batch[0], y: batch[1]})))          
      train_step.run(feed_dict = {x: batch[0], y: batch[1]})
    
    saver.save(sess, './mnist_train/model.ckpt')
    print('Test accuracy = %f' % accuracy.eval(feed_dict = {x: input.test.images, y: input.test.labels}))

def main(argv=None):
  mnist = input_data.read_data_sets('mnist_data', one_hot=True)
  train(mnist)
  return
  
if __name__ == '__main__':
  tf.app.run()
