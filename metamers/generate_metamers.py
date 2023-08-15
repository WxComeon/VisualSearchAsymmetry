import os,sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import lpips
import PIL.Image
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
from vs_model.ecc_net import load_eccNET
from vgg16_training.image_data import image_dataset_from_directory
from metamers.utils import get_layer_names, select_one_image, invert_preprocessing, feature_extraction, str2bool, CustomLearningRateSchedule, metamer_loss, postprocess_image
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Generate metamers')
parser.add_argument('--model_dir', type=str, default='/engram/nklab/wg2361/eccNET/pretrained_model', help='directory of pretrained model')
parser.add_argument('--model_types', type=str, nargs="+", default=['eccNET', 'baseline'], help='eccNET or baseline')
parser.add_argument('--data_dir', type=str, default='/share/data/imagenet/raw-data/train')
parser.add_argument('--i_batch', type=int, required=False)
parser.add_argument('--batch_size', type=int, default=20, help='batch size')
parser.add_argument('--image_size', type=int, default=224, help='image size')
parser.add_argument('--layer_num', type=int, help='layer selection')
parser.add_argument('--desired_label', type=int, required=False, help='desired label')
parser.add_argument('--visualize', type=str2bool, default=False, help='whether to visualize the generated metamers')
args = parser.parse_args()

model_types = args.model_types
im_size = args.image_size
batch_size = args.batch_size if not args.visualize else 1
visualize = args.visualize
model_dir = args.model_dir

print("loading models...")
physical_devices = tf.config.list_physical_devices('GPU')
for dev in physical_devices:
    tf.config.experimental.set_memory_growth(dev, True)
    print(dev, tf.config.experimental.get_memory_growth(dev))

models = []
for model_type in model_types:
    if model_type == 'eccNET':
        eccParam = {}
        eccParam['rf_min'] = [2]*5
        eccParam['stride'] = [2]*5
        eccParam['ecc_slope'] = [0, 0, 3.5*0.02, 8*0.02, 16*0.02]
        eccParam['deg2px'] = [round(30.0), round(30.0/2), round(30.0/4), round(30.0/8), round(30.0/16)]
        eccParam['fovea_size'] = 4
        eccParam['rf_quant'] = 1
        eccParam['pool_type'] = 'avg'
        ecc_depth = 5

        model_path = os.path.join(model_dir, "vgg16_imagenet_filters.h5")
        with tf.device('/GPU:0'):
            _, model = load_eccNET(model_path, eccParam=eccParam, ecc_depth=ecc_depth, comp_layer='diff')
        models.append(model)

    elif model_type == 'baseline':
        with tf.device('/GPU:0'):
            model = tf.keras.applications.vgg16.VGG16(weights=None)
            model.load_weights(os.path.join(model_dir, 'vgg16_baseline.h5'))
        models.append(model)

print("randomly sampling images from the dataset...")
train_ds = image_dataset_from_directory(
    args.data_dir,
    image_size=(im_size, im_size),
    crop_method='center',
    shuffle=True,
    label_mode='int',
    preserve_aspect_ratio=True,
    batch_size=batch_size)
train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE)

if not visualize:
    train_ds_subset = train_ds.take(1)
    target_images, labels = list(train_ds_subset.as_numpy_iterator())[0]
else:
    desired_label = 291
    target_images = select_one_image(desired_label, train_ds)

with torch.no_grad():
    print("initialize ipsip metric...")
    loss_fn = lpips.LPIPS(pretrained=True, net='alex', version='0.1', lpips=True, spatial=False, model_path='/engram/nklab/wg2361/lpips/weights/v0.1/alex.pth')
    loss_fn = loss_fn.to('cuda')
    
    target_images_torch = postprocess_image(target_images, device='cuda')

# optimizer learning rate schedule (Feather et al. 2022)
initial_learning_rate = 1
decay_steps = 6000
decay_rate = 0.8
num_iterations = 24000
lr_schedule = CustomLearningRateSchedule(initial_learning_rate, decay_steps, decay_rate)

# Optimization loop
df = []
with tf.device('/gpu:0'):
    for model, model_type in zip(models, model_types):
        layer_name = get_layer_names(model_type=model_type)[args.layer_num]
        feature_model = feature_extraction(model, [layer_name])
        target_features = feature_model.predict(target_images)
        assert target_features.shape[0] == batch_size
        target_layer_features = target_features.reshape(batch_size, -1)
        assert not isinstance(target_layer_features, tf.Variable)

        # Define optimizer
        random_images = np.random.randn(*target_images.shape).astype(np.float32) 
        preproccess_random_images = preprocess_input(random_images)
        metamer_images = tf.Variable(preproccess_random_images, trainable=True)
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        for i in tqdm(range(num_iterations)):
            with tf.GradientTape() as tape:
                normalized_sq_err, loss = metamer_loss(feature_model, target_layer_features, metamer_images)
            gradients = tape.gradient(loss, metamer_images)
            optimizer.apply_gradients([(gradients, metamer_images)])
            # Clip values to maintain them within [0, 255]
            metamer_images.assign(tf.clip_by_value(metamer_images, 0, 255))

            if i%1000 == 0:
                print(f"Step {i}, Loss: {loss.numpy()}")
        
        if not visualize:
            metamers = metamer_images.numpy()
            target_features = feature_model.predict(target_images)
            metamer_features = feature_model.predict(metamers)
            corr_matrix = np.corrcoef(target_features[-1].reshape(batch_size, -1), metamer_features[-1].reshape(batch_size, -1))
            corrs = corr_matrix[:batch_size, batch_size:].diagonal()
            # LPIPS measure
            with torch.no_grad():
                metamers_torch = postprocess_image(metamers, device='cuda')
                dists = loss_fn.forward(target_images_torch, metamers_torch, normalize=True).squeeze().cpu().numpy()
            
            df.append(pd.DataFrame({
                'model_type': [model_type]*batch_size,
                'layer_name': [layer_name]*batch_size,
                'image_label': labels.tolist(),
                'normalized_square_error': normalized_sq_err.numpy().tolist(),
                'correlation': corrs.tolist(),
                'lpips': dists.tolist()
            }))
            import pdb; pdb.set_trace()
        else:
            raise NotImplementedError

df = pd.concat(df)
df.to_csv(os.path.join(args.save_dir, f"metamer_lpips_layer_{args.layer_num}_batch_{args.i_batch}.csv"), index=False)