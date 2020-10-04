import onnx
import onnx.optimizer

src_onnx = 'mnist.onnx'
opt_onnx = 'mnist.opt.onnx'

# load model
model = onnx.load(src_onnx)

# optimize
model = onnx.optimizer.optimize(model, ['accuracy:0'] )

# save optimized model
with open(opt_onnx, "wb") as f:
    f.write(model.SerializeToString())

