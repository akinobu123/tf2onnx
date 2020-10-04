import time
start_time = time.time()

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
end_time_for_load_mnist = time.time()

import onnxruntime
import numpy as np
end_time_for_load_pf = time.time()

def main():
	# Tiny YOLOv2 の学習済みモデルからセッションを作る
	model = onnxruntime.InferenceSession('mnist.onnx')
	end_time_for_import_model = time.time()

	# 入力名と出力名を取得する
	input_name1 = model.get_inputs()[0].name	# 'y_' が得られるはず
	input_name2 = model.get_inputs()[1].name	# 'x' が得られるはず
	output_name = model.get_outputs()[0].name	# 'accuracy' が得られるはず

	in1 = np.array(mnist.test.images, dtype=np.float32)
	in2 = np.array(mnist.test.labels, dtype=np.float32)

	# 推論を実行する
	for i in range(30):
		output = model.run([output_name], {input_name2: in1, input_name1: in2})
		print('accuracy = ', output[0])
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
	