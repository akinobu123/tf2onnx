
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

import tensorflow as tf
from tensorflow.python.framework import graph_util

def train_and_save():
	x = tf.placeholder(tf.float32, [None, 784], name='x')
	y_ = tf.placeholder(tf.float32, [None, 10], name='y_')

	W1 = tf.Variable(tf.random_normal(shape=[784, 256], stddev=0.1))
	b1 = tf.Variable(tf.constant(0.1, shape=[256]))
	W2 = tf.Variable(tf.random_normal(shape=[256, 128], stddev=0.1))
	b2 = tf.Variable(tf.constant(0.1, shape=[128]))
	W3 = tf.Variable(tf.random_normal(shape=[128, 10], stddev=0.1))
	b3 = tf.Variable(tf.constant(0.1, shape=[10]))
	x2 = tf.nn.relu(tf.matmul(x,  W1) + b1)
	x3 = tf.nn.relu(tf.matmul(x2, W2) + b2)
	y  = tf.nn.softmax(tf.matmul(x3, W3) + b3, name='y')

	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

	cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
	train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())
		max_steps = 10000
		for step in range(max_steps):
			batch_xs, batch_ys = mnist.train.next_batch(100)
			sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
			if (step % 100) == 0:
				print(step, sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
		print(max_steps, sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

		minimal_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['y', 'accuracy'])
		tf.train.write_graph(minimal_graph, './', 'mnist.pb',  as_text=False)
		tf.train.write_graph(minimal_graph, './', 'mnist.pbtxt', as_text=True)
	return

def main():
	graph = tf.Graph()
	with graph.as_default():
		train_and_save()
	return

if __name__ == '__main__':
	main()
