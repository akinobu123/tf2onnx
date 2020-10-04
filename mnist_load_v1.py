import time
start_time = time.time()

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
end_time_for_load_mnist = time.time()

import tensorflow as tf
end_time_for_load_pf = time.time()

def import_graph_def():
	with open('mnist.pb', 'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())
		tf.import_graph_def(graph_def, name='')
	return

def show():
	print('=' * 60)
	for op in tf.get_default_graph().get_operations():
		print(op.name)
		for output in op.outputs:
			print('  ', output.name)
	print('=' * 60)
	return

def test():
	with tf.Session() as sess:
		print('accuracy = ', sess.run('accuracy:0', feed_dict={'x:0': mnist.test.images, 'y_:0': mnist.test.labels}))
	return

def main():
	graph = tf.Graph()
	with graph.as_default():
		import_graph_def()
		end_time_for_import_model = time.time()
		#show()
		for i in range(30):
			test()
		end_time_for_exec_test = time.time()

		print("load mnist :", (end_time_for_load_mnist - start_time))
		print("load platform :", (end_time_for_load_pf - end_time_for_load_mnist))
		print("import model :", (end_time_for_import_model - end_time_for_load_pf))
		print("exec test :", (end_time_for_exec_test - end_time_for_import_model))
		print("--------------------------------")
		print("total :", (end_time_for_exec_test - start_time))
	return

if __name__ == '__main__':
	main()
	