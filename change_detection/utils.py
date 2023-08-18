import os,sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import PIL.Image
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from tensorflow.keras.applications.vgg16 import preprocess_input
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from metamers.utils import invert_preprocessing

def check_label_order(labels):
    one_set = [label.split("_")[:3] for label in labels[::2]]
    other_set = [label.split("_")[:3] for label in labels[1::2]]
    if one_set == other_set:
        return True
    else:
        return False
    
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    # https://stackoverflow.com/a/312464
    for i in range(0, len(lst), n):
        yield lst[i : i + n]

def load_meta_info(meta_file_path):
    meta_info = pd.read_excel(meta_file_path, header=1, index_col=0)
    columns_with_nan = meta_info.columns[meta_info.isna().any(axis=0)]
    meta_info = meta_info.drop(columns = columns_with_nan)
    meta_info_columns = [col.strip() for col in meta_info.columns]
    meta_info.columns = meta_info_columns
    
    return meta_info

def load_stims(stim_paths, stim_dir, meta_info, crop=True, stim_size=None, out_shape=None, binary=False):
    if crop:
        crop_left = (stim_size[1] - stim_size[0])//2
        crop_right = stim_size[1] - crop_left
    stims = []
    labels = []
    for stim_path in tqdm(stim_paths):
        im = PIL.Image.open(os.path.join(stim_dir, stim_path))
        
        stim_id = int(stim_path.split("_")[0])
        if stim_path.split("_")[1] == 'L':
            x = meta_info.loc[stim_id, "X_Left"]
        else:
            assert stim_path.split("_")[1] == 'R'
            x = meta_info.loc[stim_id, "X_Right"]
            
        if crop and x < crop_left or x > crop_right:
            continue
            
        if crop:
            im = im.crop(box = (crop_left, 0, crop_right, stim_size[0]))
        if out_shape is not None:
            im = im.resize(out_shape)
        if binary:
            im = im.convert('1')
            
        stims.append(np.array(im))
        labels.append(stim_path)
        
    stims = np.stack(stims)
    return stims, labels

def get_property(labels, meta_info, data_type="ecc"):
    property_array = np.array([meta_info.loc[int(label.split("_")[0]), f"{data_type}_Left"] if label.split("_")[1] == 'L' else meta_info.loc[int(label.split("_")[0]), f"{data_type}_Right"] for label in labels])
    return property_array

def plot_model_performance(x, y_baseline, y_eccNET, x_label, y_label, title=None):
    fig, ax = plt.subplots(1,1, figsize=(6,6))
    ax.scatter(x, y_baseline)
    m, b = np.polyfit(x, y_baseline, 1)
    ax.plot(np.unique(x), m*np.unique(x) + b, label=f"baseline fit: r={np.corrcoef(x, y_baseline)[0,1]:.2f}")
            
    # eccNET
    ax.scatter(x, y_eccNET)
    m, b = np.polyfit(x, y_eccNET, 1)
    ax.plot(np.unique(x), m*np.unique(x) + b, label=f"eccNET fit: r={np.corrcoef(x, y_eccNET)[0,1]:.2f}")

    ax.legend()
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if title is not None:
        ax.set_title(title)
    
    return fig

def visualize_ecc_dependent_pooling(model, stim):
    images = []
    for i_layer in range(1,5):
        pool_layer = model.get_layer(f'Layer{i_layer}')
    #     pool_layer.compute_output_shape(input_shape=eccNET.input_shape)
        pool_layer.build(input_shape=tf.TensorShape(model.input_shape))
        input_tensor = Input(shape=model.input_shape[1:])
        output_tensor = pool_layer(input_tensor)
        model = Model(inputs=input_tensor, outputs=output_tensor)
        images.append(invert_preprocessing(model(stim).numpy()))
    return images

def euclidean_dist(x, y, normalize=True, axis=1):
    dists = np.sqrt(np.sum((x-y)**2, axis=axis))
    if normalize:
        dists = (dists-dists.min())/(dists.max()-dists.min())
    return dists

def map_label_to_mask_path(label):
    if "filler" in label:
        mask_path = "_".join(label.split("_")[:3])+"_mask.png"
    elif "mirror" in label:
        mask_path = label.split(".")[0]+"_mask.png"
    elif "window" in label:
        mask_path = "_".join(label.split("_")[:4])+"_mask.png"
    return mask_path

def resize_image_ratio(dist2screen=50, model_px2dva_unit=30):
    """Calculate the ratio for resizing task image to match with model visual degree angle and RF size."""
    cm2px = 37.8 # cm to pixels
    task_unit_in_cm = np.tan(np.deg2rad(1)) * dist2screen
    task_unit_in_px = task_unit_in_cm * cm2px
    image_ratio = model_px2dva_unit / task_unit_in_px
    return image_ratio

def pixel2vda(pixel_dist, px2vda_unit=30):
    # in eccNET model unit, 30 px/vda
    return int(pixel_dist / px2vda_unit)

def calc_triangle_sides(X_Y_ratio, vda, px2dva_unit=30):
    hypotenuse = vda * px2dva_unit
    coef = (X_Y_ratio**2+1)
    X = (hypotenuse**2/coef)**0.5
    Y = X_Y_ratio * X
    
    return X, Y

def get_fixation_point(stim_label, meta_info, stim_size, old_stim_size, crop_left, vda, px2dva_unit=30):
    """get fixation point given visual degree angle
    """
    
    stim_id = int(stim_label.split("_")[0])
    stim_side = "Right" if stim_label.split("_")[1] == 'R' else "Left"
    X_max, Y_max = stim_size
    X, Y = meta_info.loc[stim_id, f"X_{stim_side}"], meta_info.loc[stim_id, f"Y_{stim_side}"]
    X = X-crop_left
    
    resize_ratio = stim_size[0] / old_stim_size
    X, Y = X*resize_ratio, Y*resize_ratio
    X_sign, Y_sign = np.sign(X - X_max//2), np.sign(Y_max//2 - Y)
    
    if X_sign > 0 and Y_sign > 0:
        X_end, Y_end = 0, Y_max
    elif X_sign < 0 and Y_sign > 0:
        X_end, Y_end = (X_max, Y_max)
    elif X_sign < 0 and Y_sign < 0:
        X_end, Y_end = (X_max, 0)
    else:
        X_end, Y_end = (0, 0)
        
    X_dist, Y_dist = np.abs(X-X_end), np.abs(Y-Y_end)
    X_Y_ratio = Y_dist/X_dist
    rel_X, rel_Y = calc_triangle_sides(X_Y_ratio, vda=vda, px2dva_unit=px2dva_unit)
    
    if X_sign > 0 and Y_sign > 0:
        fixation = (X-rel_X, Y+rel_Y)
    elif X_sign < 0 and Y_sign > 0:
        fixation = (X+rel_X, Y+rel_Y)
    elif X_sign < 0 and Y_sign < 0:
        fixation = (X+rel_X, Y-rel_Y)
    else:
        fixation = (X-rel_X, Y-rel_Y)
    assert fixation[0] > 0 and fixation[0] < X_max and fixation[1] > 0 and fixation[1] < Y_max, "Fixation point out of image range."

    return tuple(np.array(fixation).astype(int))

def generate_and_preprocess_temp_stim(stims, eye_res, preprocess=True, bg_color=0):
    """augment stimulus with a mask and preprocess"""
    num_stims = stims.shape[0]
    num_channels = stims.shape[-1]
    stim_size = stims.shape[1:-1]
    temp_stim_size = (num_stims, stim_size[0]+2*eye_res, stim_size[1]+2*eye_res, num_channels)
    temp_stims = np.ones(temp_stim_size) * bg_color
    temp_stims[:, eye_res:stim_size[0]+eye_res, eye_res:stim_size[1]+eye_res, :] = np.copy(stims)
    if preprocess:
        temp_stims = preprocess_input(temp_stims)
    
    return temp_stims