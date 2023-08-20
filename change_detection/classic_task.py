import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
from metamers.utils import feature_extraction, plot_model_performance
from change_detection.utils import load_meta_info, load_stims, get_property
from change_detection.utils import check_label_order, chunks
from vs_model.ecc_net import load_eccNET

data_root_dir = "/engram/nklab/wg2361/CBDatabase"
stim_dir = os.path.join(data_root_dir, "Images")
meta_file_path = os.path.join(data_root_dir, "CBDatabase_Size_Eccentricity_RT.xlsx")
model_dir = "/engram/nklab/wg2361/eccNET/pretrained_model"
model_types = ["eccNET", "baseline"]
batch_size = 5

physical_devices = tf.config.list_physical_devices("GPU")
for dev in physical_devices:
    tf.config.experimental.set_memory_growth(dev, True)
    print(dev, tf.config.experimental.get_memory_growth(dev))

meta_info = load_meta_info(meta_file_path)

stim_paths = sorted(os.listdir(stim_dir))
stim_paths = [
    stim_path for stim_path in stim_paths if stim_path.split("_")[1] != "practice"
]
assert check_label_order(stim_paths)
stims, labels = load_stims(
    stim_paths, stim_dir, meta_info, crop=True, stim_size=(768, 1024, 3)
)
stims = preprocess_input(stims)

size_property = get_property(labels[::2], meta_info, "size")
ecc_property = get_property(labels[::2], meta_info, "ecc")
RT_property = get_property(labels[::2], meta_info, "RT")

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
        with tf.device("/GPU:0"):
            _, model = load_eccNET(
                model_path,
                stimuli_shape=tuple(stims[0].shape),
                eccParam=eccParam,
                ecc_depth=ecc_depth,
                comp_layer="diff",
            )

    elif model_type == "baseline":
        with tf.device("/GPU:0"):
            model = tf.keras.applications.vgg16.VGG16(
                include_top=False, weights=None, input_shape=tuple(stims[0].shape)
            )
            model.load_weights(os.path.join(model_dir, "vgg16_baseline_no_top.h5"))
    models[model_type] = model

dists = {}
for model_type in models.keys():
    feature_model = feature_extraction(models[model_type], ["block5_conv3"])
    feature_batch = []
    for stim_batch in tqdm(chunks(stims, batch_size)):
        with tf.device("/GPU:0"):
            feature_batch.append(feature_model.predict(stim_batch))
    del model, feature_model
    model_features = np.vstack(feature_batch).reshape(len(stims), -1)
    abs_dist = model_features[::2] - model_features[1::2]
    dist = np.sum(abs_dist**2, axis=1)
    del abs_dist, model_features
    dists[model_type] = (dist - dist.min()) / (dist.max() - dist.min())

include_idx = [i_size for i_size, size in enumerate(size_property) if size < 5000]
x = ecc_property[include_idx]
y_baseline = dists["baseline"][include_idx]
y_eccNET = dists["eccNET"][include_idx]
fig = plot_model_performance(
    x,
    y_baseline,
    y_eccNET,
    x_label="eccentricity (pixels)",
    y_label="normalized model euclidean distance",
)
fig.savefig("dist_eccentricity.png", dpi=300)
