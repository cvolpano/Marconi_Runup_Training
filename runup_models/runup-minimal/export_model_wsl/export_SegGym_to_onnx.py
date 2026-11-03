import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import json
import tensorflow as tf
from doodleverse_utils.model_imports import segformer

MODEL_DIR = '/mnt/d/GitLab/runup/model'
CONFIG    = f'{MODEL_DIR}/runup_14_segformer_mad_duck_equal_finetune_molokai.json'
WEIGHTS   = f'{MODEL_DIR}/runup_14_segformer_mad_duck_equal_finetune_molokai_fullmodel.h5'
ONNX_OUT  = f'{MODEL_DIR}/runup_segformer.onnx'

# ── 1. Load TF model and extract weights ─────────────────────────────────────
with open(CONFIG) as f:
    cfg = json.load(f)

id2label = {k: str(k) for k in range(cfg['NCLASSES'])}
tf_model = segformer(id2label, num_classes=cfg['NCLASSES'])
_ = tf_model(np.zeros((1, 3, 1024, 1024), dtype=np.float32))
tf_model.load_weights(WEIGHTS)
print('TF weights loaded')

tf_weights = {w.name: w.numpy() for w in tf_model.weights}

# ── 2. Build name mapping TF → PT ────────────────────────────────────────────
def tf_name_to_pt(tf_name):
    """
    Examples:
      tf_segformer_.../segformer/encoder/patch_embeddings.0/proj/kernel:0
        → segformer.encoder.patch_embeddings.0.proj.weight
      .../layer_norm/gamma:0  → ...layer_norm.weight
      .../layer_norm/beta:0   → ...layer_norm.bias
      .../dense/kernel:0      → ...dense.weight
      .../decode_head/...     → decode_head....
    """
    # strip prefix and suffix
    name = tf_name.split('tf_segformer_for_semantic_segmentation/')[-1]
    name = name.replace(':0', '')

    # slashes → dots
    name = name.replace('/', '.')

    # TF conv/dense weight name
    name = name.replace('.kernel', '.weight')

    # layer norm
    name = name.replace('.gamma', '.weight')
    name = name.replace('.beta',  '.bias')

    # batch norm running stats
    name = name.replace('.moving_mean',     '.running_mean')
    name = name.replace('.moving_variance', '.running_var')

    # decode head linear classifiers use 'classifier' in PT
    name = name.replace('decode_head.classifier.kernel', 'decode_head.classifier.weight')

    return name

# ── 3. Load PyTorch model ─────────────────────────────────────────────────────
import torch
from transformers import SegformerForSemanticSegmentation, SegformerConfig

config = SegformerConfig.from_pretrained('nvidia/mit-b0')
config.num_labels       = cfg['NCLASSES']
config.id2label         = {k: str(k) for k in range(cfg['NCLASSES'])}
config.label2id         = {str(k): k for k in range(cfg['NCLASSES'])}
pt_model = SegformerForSemanticSegmentation(config)
pt_model.eval()
pt_state = pt_model.state_dict()

# ── 4. Transfer weights ───────────────────────────────────────────────────────
new_state = {}
unmatched_tf = []
unmatched_pt = set(pt_state.keys())

for tf_name, tf_val in tf_weights.items():
    pt_name = tf_name_to_pt(tf_name)

    if pt_name not in pt_state:
        unmatched_tf.append((tf_name, pt_name))
        continue

    pt_shape = pt_state[pt_name].shape
    tf_tensor = torch.from_numpy(tf_val)

    # Conv weights: TF is (H, W, in, out), PT is (out, in, H, W)
    if tf_tensor.ndim == 4:
        tf_tensor = tf_tensor.permute(3, 2, 0, 1)

    # Dense/Linear weights: TF is (in, out), PT is (out, in)
    elif tf_tensor.ndim == 2:
        tf_tensor = tf_tensor.T

    if tf_tensor.shape != pt_shape:
        print(f'SHAPE MISMATCH: {pt_name}  TF:{tf_tensor.shape}  PT:{pt_shape}')
        unmatched_tf.append((tf_name, pt_name))
        continue

    new_state[pt_name] = tf_tensor
    unmatched_pt.discard(pt_name)

if unmatched_tf:
    print(f'\nUnmatched TF weights ({len(unmatched_tf)}):')
    for tf_n, pt_n in unmatched_tf:
        print(f'  TF: {tf_n}  →  PT: {pt_n}')

if unmatched_pt:
    print(f'\nPT weights not filled ({len(unmatched_pt)}):')
    for n in unmatched_pt:
        print(f'  {n}')

# fill num_batches_tracked (TF has no equivalent - set to 0)
for k in list(unmatched_pt):
    if 'num_batches_tracked' in k:
        new_state[k] = torch.tensor(0, dtype=torch.long)
        unmatched_pt.discard(k)

# fill any remaining PT weights with their random init (should be none)
for k in unmatched_pt:
    print(f'WARNING: using random init for {k}')
    new_state[k] = pt_state[k]

pt_model.load_state_dict(new_state)
print(f'\nTransferred {len(new_state)} weight tensors to PyTorch model')

# ── 5. Sanity check: compare outputs ─────────────────────────────────────────
test_input = np.random.rand(1, 3, 256, 256).astype(np.float32)  # smaller for speed

tf_out  = tf_model(test_input).logits.numpy()
pt_out  = pt_model(torch.from_numpy(test_input)).logits.detach().numpy()

# TF output: (1, 2, H, W) — PT output: (1, 2, H, W)  (both channels-first)
print(f'TF  output shape: {tf_out.shape},  min={tf_out.min():.4f}, max={tf_out.max():.4f}')
print(f'PT  output shape: {pt_out.shape},  min={pt_out.min():.4f}, max={pt_out.max():.4f}')
max_diff = np.abs(tf_out - pt_out).max()
print(f'Max abs difference TF vs PT: {max_diff:.6f}')
if max_diff < 0.01:
    print('✓ Outputs match — weight transfer successful')
else:
    print('✗ Outputs differ — check unmatched weights above')

# ── 6. Export to ONNX ─────────────────────────────────────────────────────────
dummy_pt = torch.zeros(1, 3, 1024, 1024)
torch.onnx.export(
    pt_model,
    dummy_pt,
    ONNX_OUT,
    opset_version=17,
    input_names=['input'],
    output_names=['logits'],
    dynamic_axes={'input': {0: 'batch'}, 'logits': {0: 'batch'}},
    do_constant_folding=True,
)
print(f'ONNX export complete → {ONNX_OUT}')