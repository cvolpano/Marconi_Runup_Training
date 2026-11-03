import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession(
    '/mnt/d/GitLab/runup/model/runup_segformer.onnx',
    providers=['CPUExecutionProvider']
)

print('Inputs:', [(i.name, i.shape) for i in sess.get_inputs()])
print('Outputs:', [(o.name, o.shape) for o in sess.get_outputs()])

# Run a dummy inference
dummy = np.random.rand(1, 3, 1024, 1024).astype(np.float32)
result = sess.run(None, {'input': dummy})
print('Output shape:', result[0].shape)
print('Output min/max:', result[0].min(), result[0].max())
print('Any NaN:', np.isnan(result[0]).any())