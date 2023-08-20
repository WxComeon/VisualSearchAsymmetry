import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path
import itertools
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
import tensorflow as tf

from vs_model.ecc_net import load_eccNET
from metamers.utils import feature_extraction
from change_detection.utils import (
    load_meta_info,
    load_stims,
    check_label_order,
    map_label_to_mask_path,
    euclidean_dist,
    resize_image_ratio,
    pixel2vda,
    downsample_attention_map,
    get_fixation_point,
    generate_and_preprocess_model_stim,
)

argparser = argparse.ArgumentParser()
argparser.add_argument("--vda", type=int, help="visual degree of attention")
args = argparser.parse_args()

# physical_devices = tf.config.list_physical_devices('GPU')
# for dev in physical_devices:
#     tf.config.experimental.set_memory_growth(dev, True)
#     print(dev, tf.config.experimental.get_memory_growth(dev))

vda = args.vda
data_root_dir = "/engram/nklab/wg2361/CBDatabase"
stim_dir = os.path.join(data_root_dir, "Images")
mask_dir = os.path.join(data_root_dir, "Masks")
meta_file_path = os.path.join(data_root_dir, "CBDatabase_Size_Eccentricity_RT.xlsx")
model_dir = "/engram/nklab/wg2361/eccNET/pretrained_model"
model_types = ["baseline", "eccNET"]
output_dir = "/engram/nklab/wg2361/exps/cb_ecc"
Path(output_dir).mkdir(parents=True, exist_ok=True)

model_px2dva_unit = 30
image_ratio = resize_image_ratio(dist2screen=50, model_px2dva_unit=model_px2dva_unit)
old_stim_size = (768, 1024, 3)
out_shape = int(old_stim_size[0] * image_ratio)
crop_left = (old_stim_size[1] - old_stim_size[0]) // 2
crop_right = old_stim_size[1] - crop_left

meta_info = load_meta_info(meta_file_path)
stim_paths = sorted(os.listdir(stim_dir))
stim_paths = [
    stim_path for stim_path in stim_paths if stim_path.split("_")[1] != "practice"
]
stims, labels = load_stims(
    stim_paths,
    stim_dir,
    meta_info,
    crop=True,
    stim_size=old_stim_size,
    out_shape=(out_shape, out_shape),
)
assert check_label_order(labels)
labels = np.array(labels)

mask_paths = [[map_label_to_mask_path(label)] * 2 for label in labels[::2]]
mask_paths = np.array(list(itertools.chain(*mask_paths)))
mask_paths_prefix = [mask_path.split("_")[:3] for mask_path in mask_paths]
label_prefix = [label.split("_")[:3] for label in labels]
assert mask_paths_prefix == label_prefix
all_mask_paths = os.listdir(mask_dir)
assert np.all([mask_path in all_mask_paths for mask_path in mask_paths])

eye_res = out_shape
model_input_shape = (2 * eye_res, 2 * eye_res, 3)

models = {}
for model_type in model_types:
    if model_type == "eccNET":
        eccParam = {}
        eccParam["rf_min"] = [2] * 5
        eccParam["stride"] = [2] * 5
        eccParam["ecc_slope"] = [0, 0, 3.5 * 0.02, 8 * 0.02, 16 * 0.02]
        eccParam["deg2px"] = [
            round(30.0),
            round(30.0 / 2),
            round(30.0 / 4),
            round(30.0 / 8),
            round(30.0 / 16),
        ]
        eccParam["fovea_size"] = 4
        eccParam["rf_quant"] = 1
        eccParam["pool_type"] = "avg"
        ecc_depth = 5

        model_path = os.path.join(model_dir, "vgg16_imagenet_filters.h5")
        with tf.device("/CPU:0"):
            _, model = load_eccNET(
                model_path,
                stimuli_shape=model_input_shape,
                eccParam=eccParam,
                ecc_depth=ecc_depth,
                comp_layer="diff",
            )

    elif model_type == "baseline":
        with tf.device("/CPU:0"):
            model = tf.keras.applications.vgg16.VGG16(
                include_top=False, weights=None, input_shape=model_input_shape
            )
            model.load_weights(os.path.join(model_dir, "vgg16_baseline_no_top.h5"))
    models[model_type] = model

# load the attention maps at the original dimension, and background mask before resizing.
attention_maps, attention_labels = load_stims(
    mask_paths,
    mask_dir,
    meta_info,
    crop=True,
    stim_size=old_stim_size,
    out_shape=(out_shape, out_shape),
    binary=True,
)
attention_maps = 1 - attention_maps
attention_maps = np.expand_dims(attention_maps.astype(int), axis=-1)
attention_size={}
for model_type in model_types:
    attention_size[model_type] = models[model_type].get_layer("block5_conv3").output_shape[1:-1]

dfs = []
for model_type in models.keys():
    print(model_type)
    feature_model = feature_extraction(models[model_type], ["block5_conv3"])
    feature_batch = []
    for i_stim, stim in tqdm(enumerate(stims)):
        stim = np.expand_dims(stim, axis=0)
        attention_map = np.expand_dims(attention_maps[i_stim], axis=0)
        stim_size = stim.shape[1:-1]
        X_fixation, Y_fixation = get_fixation_point(
            labels[i_stim],
            meta_info,
            stim_size,
            old_stim_size=old_stim_size[0],
            crop_left=crop_left,
            vda=vda,
            px2dva_unit=30,
        )
        model_stim = generate_and_preprocess_model_stim(
            stim, eye_res=eye_res, preprocess=True, bg_color=0
        )
        model_stim = (model_stim)[:, Y_fixation:Y_fixation+eye_res*2, X_fixation:X_fixation+eye_res*2, :]
        model_attention = generate_and_preprocess_model_stim(
            np.expand_dims(attention_maps[i_stim], axis=0),
            eye_res=eye_res,
            preprocess=False,
            bg_color=0,
        )
        model_attention = (model_attention)[:, Y_fixation:Y_fixation+eye_res*2, X_fixation:X_fixation+eye_res*2, :]
        model_attention = downsample_attention_map(model_attention, attention_size[model_type])
        
        with tf.device("/CPU:0"):
            model_features = feature_model.predict(model_stim)
            model_features = model_features * model_attention
            feature_batch.append(model_features.reshape(1,-1))
    feature_batch = np.concatenate(feature_batch, axis=0)
    dists = euclidean_dist(feature_batch[::2], feature_batch[1::2], axis=1, normalize=False)
    del feature_batch
    df = pd.DataFrame({
        'model_type':[model_type]*len(dists),
        'eccentricity': [vda] * len(dists),
        'stim_label': labels[::2],
        'model_dist': dists,
    })
    dfs.append(df)
df = pd.concat(dfs, axis=0)
df.to_csv(os.path.join(output_dir, f"dist_{vda}_deg.csv"))