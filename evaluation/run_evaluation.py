import numpy as np
import tensorflow as tf

import lpips_tf
import elpips

import os, io
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse
import glob

import cv2
import imageio
from PIL import Image, ImageCms
from skimage import exposure

from skimage.metrics import structural_similarity

def mse_to_psnr(mse):
  """Compute PSNR given an MSE (we assume the maximum pixel value is 1)."""
  return -10.0 / np.log(10.0) * np.log(mse)

def compute_psnr(image0, image1):
    mse = ((image0 - image1) ** 2).mean()
    return -10.0 / np.log(10.0) * np.log(mse)

def compute_ssim(image0, image1):
    return structural_similarity(image1, image0, win_size=11, multichannel=True, gaussian_weights=True, data_range=1.0)

def compute_lpips(image0_ph, image1_ph, distance_t, image0, image1, sess):
    return sess.run(distance_t, feed_dict={image0_ph: image0, image1_ph: image1})

def imread(f):
    img = imageio.imread(f)
    return img

def get_files(args):
    gt_files = sorted(glob.glob(os.path.join(args.gt_dir, '*.npy')))
    pred_files = sorted(glob.glob(os.path.join(args.pred_dir, '*.npy')))

    return gt_files, pred_files

def main(args):
    gt_files, pred_files = get_files(args)

    # Output values
    lpips_vals = []
    ssim_vals = []
    psnr_vals = []

    # E-LPIPS setup
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    image0_ph = tf.placeholder(tf.float32)
    image1_ph = tf.placeholder(tf.float32)

    with tf.Session(config=config) as session:
        if args.use_elpips:
            metric = elpips.Metric(elpips.elpips_vgg(batch_size=1))
            distance_t = metric.forward(image0_ph, image1_ph)
        else:
            distance_t = lpips_tf.lpips(image0_ph, image1_ph, model='net-lin', net='vgg')

        # Run
        for i, (gt_file, pred_file) in enumerate(zip(gt_files, pred_files)):
            print(f'Reading {gt_file} {pred_file}')

            gt_img = np.load(os.path.join(args.gt_dir, gt_file))
            pred_img = np.load(os.path.join(args.pred_dir, pred_file))

            if args.use_masks:
                mask = np.repeat(np.all((gt_img == 1.0), axis=-1, keepdims=True).reshape(-1, 1), 3, axis=-1).reshape(gt_img.shape)
                pred_img[mask] = 1.0

            # Eval
            psnr_vals.append(compute_psnr(gt_img, pred_img))
            lpips_vals.append(
                compute_lpips(image0_ph, image1_ph, distance_t, gt_img[None], pred_img[None], session)
            )
            ssim_vals.append(compute_ssim(gt_img, pred_img))

            print()
            print(f'Image {i}')
            print()
            print( "PSNR:",  psnr_vals[-1] )
            print( "SSIM:",  ssim_vals[-1] )
            print( "LPIPS:", lpips_vals[-1] )
            print()

            print( "Running Mean/Std PSNR:",  np.mean(psnr_vals ), np.std(psnr_vals ) )
            print( "Running Mean/Std SSIM:",  np.mean(ssim_vals ), np.std(ssim_vals ) )
            print( "Running Mean/Std LPIPS:", np.mean(lpips_vals), np.std(lpips_vals) )
            print()

    print("Total Mean LPIPS:", np.mean(lpips_vals))
    print("Total Mean SSIM:", np.mean(ssim_vals))
    print("Total Mean PSNR:", np.mean(psnr_vals))

    with open(
        os.path.join(args.metrics_file),
        'w'
    ) as f:
        f.write(f"LPIPS Mean, Std: {np.mean(lpips_vals)} +- {np.std(lpips_vals)}\n")
        f.write(f"SSIM Mean, Std: {np.mean(ssim_vals)} +- {np.std(ssim_vals)}\n")
        f.write(f"PSNR Mean, Std: {np.mean(psnr_vals)} +- {np.std(psnr_vals)}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--gt_dir", type=str, default='gt', help='')
    parser.add_argument("--pred_dir", type=str, default='pred', help='')
    parser.add_argument("--metrics_file", type=str, default='metrics.txt', help='')
    parser.add_argument("--use_elpips", action='store_true', help='')
    parser.add_argument("--use_masks", action='store_true', help='')

    args = parser.parse_args()

    main(args)
