"""
ONNX replacement for run_segmentation.py.
Requires only: onnxruntime, numpy, scikit-image, pillow
No TensorFlow, no doodleverse, no gym environment.
"""
import os
import argparse
import numpy as np
from glob import glob
from pathlib import Path
from datetime import datetime, timedelta

import onnxruntime as ort
from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2gray


# ── Preprocessing (mirrors doodleverse get_image + standardize) ───────────────

def standardize(img):
    """Adjusted std normalization — exact copy of doodleverse_utils.imports.standardize."""
    N = img.shape[0] * img.shape[1]
    s = np.maximum(np.std(img), 1.0 / np.sqrt(N))
    m = np.mean(img)
    img = (img - m) / s
    return img


def preprocess(img_path, target_size=1024):
    """
    Load a jpg/png timestack image and prepare it for the ONNX SegFormer.

    Pipeline mirrors doodleverse get_image() with MODEL='segformer', N_DATA_BANDS=1:
      1. Read image, convert to grayscale
      2. Resize to (target_size, target_size)
      3. standardize (zero mean, adjusted-std)
      4. Stack to 3 channels  (standardize does this for 2D input)
      5. Transpose to channels-first (3, H, W)
      6. Add batch dim → (1, 3, H, W)

    Returns:
        input_tensor : float32 array (1, 3, target_size, target_size)
        orig_h       : original image height
        orig_w       : original image width
    """
    img = imread(img_path)
    orig_h, orig_w = img.shape[:2]
    print(f'  raw shape: {img.shape}')

    # grayscale — handle 2D, 3-channel, and 4-channel inputs
    if img.ndim == 3:
        img = rgb2gray(img[:, :, :3])   # (H, W) float64 in [0,1]
    elif img.ndim == 2:
        img = img.astype(np.float64) / 255.0  # normalise to [0,1] like rgb2gray does
    img = img.astype(np.float32)
    print(f'  after grayscale: {img.shape}')

    # resize to square target
    img = resize(img, (target_size, target_size),
                 preserve_range=True, clip=True).astype(np.float32)
    print(f'  after resize: {img.shape}')

    # standardize — returns (H, W, 3) when input is 2D
    img = standardize(img).astype(np.float32)
    print(f'  after standardize: {img.shape}')

    # ensure 3 channels
    if img.ndim == 2:
        img = np.dstack((img, img, img))

    # channels-first → (3, H, W)
    img = np.transpose(img, (2, 0, 1))

    # batch dim → (1, 3, H, W)
    return img[np.newaxis], orig_h, orig_w


# ── Postprocessing (mirrors doodleverse est_label_binary + do_seg) ────────────

def postprocess(logits, orig_h, orig_w, target_size=1024):
    """
    Convert raw SegFormer logits to the av_prob_stack saved by the original pipeline.

    Mirrors doodleverse for NCLASSES=2:
      logits : (1, 2, H/4, W/4)  — SegFormer outputs at 1/4 resolution
      1. Squeeze batch dim
      2. Resize each class channel to (target_size, target_size)
      3. Transpose to (target_size, target_size, 2)   channels-last
      4. Resize to (orig_h, orig_w, 2)
      5. Compute av_prob_stack = (e1 + (1 - e0)) / 2  — same formula as do_seg

    Returns:
        av_prob_stack : float32 array (orig_h, orig_w)  — drop-in for getSoftMaxScores()
    """
    est = logits[0].astype(np.float32)                          # (2, H/4, W/4)

    # resize to target_size then transpose to channels-last
    est = resize(est, (2, target_size, target_size),
                 preserve_range=True, clip=True)                # (2, 1024, 1024)
    est = np.transpose(est, (1, 2, 0))                         # (1024, 1024, 2)

    # resize to original image dimensions
    est = resize(est, (orig_h, orig_w),
                 preserve_range=True, clip=True).astype(np.float32)  # (H, W, 2)

    e0 = est[:, :, 0]
    e1 = est[:, :, 1]
    av_prob_stack = (e1 + (1 - e0)) / 2
    return av_prob_stack


# ── Main ──────────────────────────────────────────────────────────────────────

def run_segmentation_onnx(site_name):
    from runup.pixel2runup import getSite
    config_dir = str(Path(__file__).resolve().parent.parent / 'config')
    site = getSite(site_name, config_dir)

    onnx_path   = str(Path(__file__).resolve().parent.parent / 'models' / 'runup_segformer.onnx')
    sample_dir  = site.segDir
    start_date  = site.startDate
    end_date    = site.endDate
    target_size = 1024

    print(f'Loading ONNX model: {onnx_path}')
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    sess = ort.InferenceSession(onnx_path, providers=providers)
    input_name = sess.get_inputs()[0].name
    print(f'Using provider: {sess.get_providers()}')

    sample_files = sorted(glob(os.path.join(sample_dir, '*.jpg')))
    if not sample_files:
        sample_files = sorted(glob(os.path.join(sample_dir, '*.png')))
    print(f'Found {len(sample_files)} images in {sample_dir}')

    for f in sample_files:
        filename      = os.path.basename(f)
        base          = os.path.splitext(filename)[0]
        npy_out       = os.path.join(sample_dir, base + '_softmax.npy')

        if os.path.exists(npy_out):
            print(f'Already exists, skipping: {filename}')
            continue

        epoch_time = int(filename.split('.')[0])
        from datetime import timezone
        file_date = datetime.fromtimestamp(epoch_time, tz=timezone.utc)
        if file_date < start_date or file_date > end_date + timedelta(days=1):
            continue

        print(f'Processing: {filename}')
        try:
            inp, orig_h, orig_w = preprocess(f, target_size)
            logits = sess.run(None, {input_name: inp})[0]       # (1, 2, 256, 256)
            av_prob_stack = postprocess(logits, orig_h, orig_w, target_size)
            np.save(npy_out, av_prob_stack)
            os.remove(f)
        except Exception as e:
            print(f'  Failed: {e}')

    print('Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run ONNX SegFormer segmentation on timestacks (no TensorFlow required)')
    parser.add_argument('site_name', help='Site name matching config file (e.g. MadBeach2022)')
    args = parser.parse_args()
    run_segmentation_onnx(args.site_name)