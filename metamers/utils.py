
import copy
import itertools
import numpy as np
import torch
import tensorflow as tf

def get_layer_names(model_type):
    blocks = [[1,2], [3,4], [5]]
    convs = [[2], [3], [3]]

    layer_names = []
    for i_block, block in enumerate(blocks):
        for layer in block:
            this_layer_names = [f'block{layer}_conv{conv}' for conv in range(1, convs[i_block][0]+1)]
            if len(block)>1:
                if model_type == 'eccNET':
                    layer_names.append(this_layer_names + [f'Layer{layer}'])
                elif model_type == 'baseline':
                    layer_names.append(this_layer_names + [f'block{layer}_pool'])
            else:
                layer_names.append(this_layer_names)
    layer_names = list(itertools.chain(*layer_names))
    
    return layer_names

def select_one_image(desired_label, preprocessed_train_ds):
    """select one image from the network based on one specific label"""
    preprocessed_train_ds = preprocessed_train_ds.prefetch(tf.data.experimental.AUTOTUNE)
    unbatch_preprocessed_train_ds = preprocessed_train_ds.unbatch()
    image_with_desired_label = unbatch_preprocessed_train_ds.filter(lambda image, label: tf.equal(label, desired_label)).take(1)
    target_image, _ = list(image_with_desired_label.as_numpy_iterator())[0]
    target_image = np.expand_dims(target_image, axis=0)
    return target_image

def invert_preprocessing(image):
    # Add back the mean values for the BGR channels
    image = copy.deepcopy(image)
    image[..., 0] += 103.939
    image[..., 1] += 116.779
    image[..., 2] += 123.68

    # Convert from BGR to RGB
    image = image[..., ::-1]

    # Ensure the values are in the valid range
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image

def feature_extraction(model, layer_names):
    """extract layer features from the model"""
    layer_outputs = [model.get_layer(name).output for name in layer_names]
    feature_extraction_model = tf.keras.Model(inputs=model.input, outputs=layer_outputs)
    return feature_extraction_model

def str2bool(v):  # https://stackoverflow.com/a/43357954
    from argparse import ArgumentTypeError

    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError("Boolean value expected.")

class CustomLearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, decay_steps, decay_rate):
        super(CustomLearningRateSchedule, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        
    def __call__(self, step):
        # Compute how many times the learning rate has decayed
        decay_count = step // self.decay_steps
        return self.initial_learning_rate * (self.decay_rate ** decay_count)
    
def metamer_loss(feature_extraction_model, target_layer_features, cur_imgs):
    """compute the metamer loss as normalized squared error between the target and current layer features"""
    batch_size = target_layer_features.shape[0]
    target_norm = tf.norm(target_layer_features, ord='euclidean', axis=-1)
    
    cur_layer_features = feature_extraction_model(cur_imgs)
    cur_layer_features = tf.reshape(cur_layer_features, (batch_size, -1))
    
    feature_diff = target_layer_features - cur_layer_features
    feature_diff_norm = tf.norm(feature_diff, ord='euclidean', axis=-1)
    normalized_sq_err = tf.math.divide(feature_diff_norm, target_norm)
    
    return normalized_sq_err, tf.math.reduce_mean(normalized_sq_err)

def postprocess_image(image, device):
    """convert from eccNET to LPIPS input format"""
    image = invert_preprocessing(image)/255.
    image_torch = torch.tensor(image, dtype=torch.float32, device=device).permute(0,3,1,2)
    return image_torch