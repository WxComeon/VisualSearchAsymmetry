import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda

@tf.custom_gradient
def relu_with_custom_gradient(x):
    def grad(dy):
        return tf.ones_like(dy)
    return tf.nn.relu(x), grad

def custom_vgg16(block_num):
    # Ensure block number is valid
    assert block_num in [1, 2, 3, 4, 5], "Invalid block number"
    num_layers = [2, 2, 3, 3, 3]

    # Load pretrained VGG16 model
    base_model = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    base_weights = base_model.get_weights()

    # Helper function to build a VGG block
    def vgg_block(x, filters, block_name, custom_relu=False):
        block_num_layers = num_layers[block_name-1]
        for i in range(block_num_layers-1):
            x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same', name=f'block{block_name}_conv{i+1}', activation='relu')(x)
        if custom_relu:
            x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same', name=f'block{block_name}_conv{block_num_layers}')(x)
            x = Lambda(relu_with_custom_gradient if custom_relu else tf.nn.relu)(x)
        else:
            x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same', name=f'block{block_name}_conv{block_num_layers}', activation='relu')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name=f'block{block_name}_pool')(x)
        return x

    # Start building the model
    input_tensor = Input(shape=(224, 224, 3))
    x = input_tensor

    for i, filters in enumerate([64, 128, 256, 512, 512]):
        x = vgg_block(x, filters, i+1, custom_relu=(i+1 == block_num))

    custom_model = Model(input_tensor, x)
    custom_model.set_weights(base_weights)

    return custom_model

# Example: Customize the gradient for block 3
model = custom_vgg16(3)