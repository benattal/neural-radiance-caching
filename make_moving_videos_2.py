import numpy as np 
import os
import tqdm 
import h5py
from glob import glob
import skimage.io
import cv2
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import gaussian_filter
import matplotlib


matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def scale_value_hsv(image, mask):
  """
  Applies a multiplicative mask to an image, preserving saturation and hue.

  Args:
      image: A 3D numpy array representing the RGB image.
      mask: A 2D numpy array representing the multiplicative mask.

  Returns:
      A 3D numpy array representing the modified image.
  """
  # Ensure mask has the same shape as the image's first two dimensions
  mask = np.broadcast_to(mask, (image.shape[0], image.shape[1]))

  # Convert the image to HSV color space
  hsv_image = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_RGB2HSV)

  # Apply mask to the value channel only
  hsv_image[..., 2] *= mask.astype(np.float32)

  # Clip values to ensure they stay within valid HSV range (0-1)
  hsv_image = np.clip(hsv_image, 0, 1)

  # Convert back to BGR color space
  modified_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

  return modified_image


def apply_3d_gaussian_filter(volume, sigma):
  """
  Applies a 3D Gaussian filter to a 3D volume.

  Args:
      volume: A 3D numpy array representing the 3D volume.
      sigma: A tuple (sigma_x, sigma_y, sigma_z) representing the standard deviations
              of the Gaussian filter for each dimension.

  Returns:
      A 3D numpy array representing the filtered volume.
  """
  # Ensure sigma is a tuple of length 3
  if not isinstance(sigma, tuple) or len(sigma) != 3:
    raise ValueError("Sigma must be a tuple of length 3 (sigma_x, sigma_y, sigma_z).")

  # Apply Gaussian filter along each dimension using mode='same' for centered filtering
  return gaussian_filter(volume, sigma)


def apply_gaussian_filter(volume, sigma):
  """
  Applies a Gaussian filter to a 3D volume only along the time dimension
  without using for loops.

  Args:
      volume: A 4D numpy array representing the 3D volume with the time dimension
              as the last dimension.
      sigma: The standard deviation of the Gaussian filter for the time dimension.

  Returns:
      A 4D numpy array with the filtered volume.
  """
  # Reshape the volume to have spatial dimensions flattened and time as the last dimension
  reshaped_volume = volume.reshape(-1, volume.shape[-1])
  # Apply Gaussian filter along the last dimension (time) using broadcasting
  filtered_volume = gaussian_filter1d(reshaped_volume, sigma, axis=-1)
  # Reshape back to the original shape
  return filtered_volume.reshape(volume.shape)


def gray_world(img, scales=None, get_scales=False):
    """Applies the Gray World white balancing algorithm."""
    r, g, b = cv2.split(img)  # Split the image into BGR channels
    
    if scales is not None:
        scale_r = scales[0]
        scale_b = scales[1]
        scale_g = scales[2]
    else:
        # Calculate average values for each channel
        avg_b = np.mean(b)
        avg_g = np.mean(g)
        avg_r = np.mean(r)

        # Calculate the overall average
        avg_gray = (avg_r + avg_g + avg_b) / 3 

        # Calculate the scaling factors
        scale_r = avg_gray / avg_r
        scale_g = avg_gray / avg_g
        scale_b = avg_gray / avg_b

    # Scale each channel to compensate for color cast
    result = np.zeros(img.shape, img.dtype)
    result[:, :, 2] = b * scale_b
    result[:, :, 1] = g * scale_g
    result[:, :, 0] = r * scale_r

    if get_scales:
        return [scale_r, scale_b, scale_g]

    # Ensure that pixel values don't exceed the maximum (important!)
    result = np.clip(result, 0, 255).astype('uint8')
    return result


def sharpen_color_image(image, alpha=2.0, color=True):
    """Sharpens a color image using Laplacian filtering."""

    if not color:
        laplacian = cv2.Laplacian(image, cv2.CV_32F)
        laplacian = laplacian.astype('uint8')
        return cv2.addWeighted(image, 1.0, laplacian, alpha, 0)

    
    # Apply Laplacian filter directly to each color channel
    laplacian_b = cv2.Laplacian(image[:, :, 0], cv2.CV_64F)
    laplacian_b = laplacian_b.astype('uint8')
    laplacian_g = cv2.Laplacian(image[:, :, 1], cv2.CV_64F)
    laplacian_g = laplacian_g.astype('uint8')
    laplacian_r = cv2.Laplacian(image[:, :, 2], cv2.CV_64F)
    laplacian_r = laplacian_r.astype('uint8')

    # Combine sharpened channels back into color image
    sharpened_b = cv2.addWeighted(image[:, :, 0], 1.0, laplacian_b, alpha, 0)
    sharpened_g = cv2.addWeighted(image[:, :, 1], 1.0, laplacian_g, alpha, 0)
    sharpened_r = cv2.addWeighted(image[:, :, 2], 1.0, laplacian_r, alpha, 0)
    sharpened_image = cv2.merge([sharpened_b, sharpened_g, sharpened_r])

    return sharpened_image


def increase_saturation_hsv(image, factor=1.5):
  """Increases the saturation of an image using HSV color space.

  Args:
      image: The input image in BGR format.
      factor: The factor by which to increase saturation (default: 1.5).

  Returns:
      The image with increased saturation in BGR format.
  """
  hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
  hsv_img[..., 1] = np.clip(hsv_img[..., 1] * factor, 0, 255)
  return cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)


def read_h5(path, load_clip=None):
    with h5py.File(path, 'r') as f:
        if load_clip is not None:
            frames = np.array(f['data'][:, :, load_clip[0]:load_clip[1]])
        else:
            frames = np.array(f['data'])
    return frames


def normalize(img):
    return (img - img.min()) / (img.max() - img.min())




names = {"direct": "cache_direct_color", 
         "indirect": "cache_indirect_color", 
         "transients":"cache_time_slice", 
         "intensity": "color_cache0"}

def extract_transient_frames(path, composited=True, bkgd=None, gamma=2, scale=1, img_gamma=2, img_scale=1,
                             alpha=0.9, fade_range=(20, -20), add_timer=False, fade_num=5, out_folder='transient_images', time_slice=0, time_range=1, time_mult=1):

    
    
    if "transient" in out_folder:
        target_folder = "transients"
        extension = "*.npy"
    elif "time_slice" in out_folder:
        target_folder = "transients"
        extension = "*.h5"
    elif "indirect" in out_folder:
        target_folder = "cache_indirect_color"
        extension = "*.png"
    elif "direct" in out_folder:
        target_folder = "cache_direct_color"
        extension = "*.png"
    elif "intensity" in out_folder:
        target_folder = "color_cache0"
        extension = "*.png"


    file_fnames = sorted(glob(os.path.join(path, 'save', target_folder, extension )))
    fade_range = (fade_range[0], len(file_fnames) + fade_range[1])

    if composited:
        img_fnames = sorted(glob(os.path.join(bkgd, "save", "color_cache0" , f'*.png')))

    for ind, file_fname in tqdm(enumerate(file_fnames)):
        if "png" in file_fname:
            img = skimage.io.imread(file_fname).astype(np.float32) / 255.
        elif "h5" in file_fname:
            img = read_h5(file_fname, load_clip=(time_slice, time_slice+time_range)).mean(axis=-2) * time_mult
        else:
            img = np.load(file_fname).squeeze()      
    
        img = np.clip((img/scale)**(gamma), 0, 1)

        if composited:
            img_bkg = skimage.io.imread(img_fnames[ind]).astype(np.float32) / 255.
            img = img.reshape(img_bkg.shape[:2] + (-1,))
            img_bkg = (img_bkg/img_scale)**(img_gamma)
            img = alpha*img + (1-alpha)*img_bkg[..., None]
        
        if "transient" in out_folder:
            out = np.zeros((img.shape[0], img.shape[1], 3))
            if "red" in out_folder:
                out[..., 0] = img
            if "green" in out_folder:
                out[..., 1] = img
            if "blue" in out_folder:
                out[..., 2] = img
            img = out
        
            if ind<fade_range[0]:
                s = 0.5 + ind/(2*fade_range[0])

            elif ind>fade_range[1]:
                s = ((len(file_fnames)-ind)/(len(file_fnames) - fade_range[1]))/2 + 0.5

            img = s*img
    
        out = (255*np.clip(img, 0, 1)).astype(np.uint8).squeeze()
        
        os.makedirs(out_folder, exist_ok=True)
        fname = os.path.join(out_folder, os.path.basename(file_fname)[:-4] + f'.png')
        skimage.io.imsave(fname, out)






def combine_into_rgb(path, folders, channels=["red", "green", "blue"]):
    
    for folder in folders:
        fnames_red = sorted(glob(os.path.join(f"{path}_{channels[0]}_{folder}", '*.png')))
        fnames_green = sorted(glob(os.path.join(f"{path}_{channels[1]}_{folder}", '*.png')))
        fnames_blue = sorted(glob(os.path.join(f"{path}_{channels[2]}_{folder}", '*.png')))
    
        for i1p, i2p, i3p in tqdm(zip(fnames_red, fnames_green, fnames_blue), total=len(fnames_red)):
            i1 = skimage.io.imread(i1p)
            i2 = skimage.io.imread(i2p)
            i3 = skimage.io.imread(i3p)
            if len(i1.shape)>2:
                i1[..., 1:] = 0 
                i2[..., [0, 2]] = 0 
                i3[..., :2] = 0 
                comb = i1+i2+i3
            else:
                comb = np.stack([i1, i2, i3], -1)
            
            fname = os.path.join(f"{path}_rgb_{folder}", os.path.basename(i1p))
            os.makedirs(f"{path}_rgb_{folder}", exist_ok=True)
            skimage.io.imsave(fname, comb)


def add_text_to_frame(img, text, font_size=16, alpha=255, text_position=(10, 10) ):
    from PIL import Image, ImageDraw, ImageFont
    image = Image.fromarray(img).convert("RGBA")
    txt = Image.new('RGBA', image.size, (0,0,0,0))
    draw = ImageDraw.Draw(txt)
    font = ImageFont.truetype("media/FuturaPTMedium.otf", font_size)
    draw.text(text_position, text, fill=(255, 255, 255, alpha), font=font)
    combined = Image.alpha_composite(image, txt)    
    return np.asarray(combined)


    


if __name__=="__main__":
    moving_folder = "./checkpoints/yobo_results/synthetic"

    queue = [ "house"]
    types = ["direct", "indirect", "transient", "intensity"]

    if "house" in queue:
        channels = ["red", "green", "blue"]
        # types = ["direct", "indirect", "transient", "intensity", "time_slice"]
        types = ["direct", "indirect", "time_slice", "intensity"]
        # have to process direct, indirect, transient, intensity 

        for i, c in enumerate(channels):
            # transient
            # if "transient" in types:
            #     extract_transient_frames(f'{moving_folder}/house_material_light_resample_finetune_{i}_eval_fixed_light_fixed_camera', bkgd=f'{moving_folder}/house_material_light_from_scratch_resample_eval_fixed_light_fixed_camera',
            #     composited=True, gamma=1/2, scale=0.2, img_scale=0.8, img_gamma=1/2, alpha=0.8, out_folder=f'../image_folders_moving/house_{c}_transient')

            if "time_slice" in types:
                extract_transient_frames(f'{moving_folder}/house_material_light_resample_finetune_{i}_eval_fixed_light_fixed_camera', bkgd=f'{moving_folder}/house_material_light_from_scratch_resample_eval_fixed_light_fixed_camera',
                # composited=True, gamma=1/2, scale=0.2, img_scale=0.8, img_gamma=1/2, alpha=0.8, out_folder=f'../image_folders_moving/house_{c}_time_slice', time_slice=550, time_range=1, time_mult=1)
                composited=True, gamma=1/2, scale=0.2, img_scale=0.8, img_gamma=1/2, alpha=0.8, out_folder=f'../image_folders_moving/house_{c}_time_slice', time_slice=750, time_range=1, time_mult=1)
            
            # if "direct" in types:
            #     extract_transient_frames(f'{moving_folder}/house_material_light_resample_finetune_{i}_eval_fixed_light_fixed_camera', bkgd=f'{moving_folder}/house_material_light_from_scratch_resample_eval_fixed_light_fixed_camera',
            #     composited=False, gamma=1, scale=1, img_scale=0.8, img_gamma=1/2, alpha=0.9, out_folder=f'../image_folders_moving/house_{c}_direct')

            # if "indirect" in types:
            #     extract_transient_frames(f'{moving_folder}/house_material_light_resample_finetune_{i}_eval_fixed_light_fixed_camera', bkgd=f'{moving_folder}/house_material_light_from_scratch_resample_eval_fixed_light_fixed_camera',
            #     composited=False, gamma=1, scale=0.5, img_scale=0.8, img_gamma=1/2, alpha=0.9, out_folder=f'../image_folders_moving/house_{c}_indirect')

            # if "intensity" in types:
            #     extract_transient_frames(f'{moving_folder}/house_material_light_resample_finetune_{i}_eval_fixed_light_fixed_camera', bkgd=f'{moving_folder}/house_material_light_from_scratch_resample_eval_fixed_light_fixed_camera',
            #     composited=False, gamma=1, scale=1, img_scale=0.8, img_gamma=1/2, alpha=0.9, out_folder=f'../image_folders_moving/house_{c}_intensity')

        combine_into_rgb('../image_folders_moving/house', types)

