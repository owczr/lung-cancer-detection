"""
Module with ResNet concrete builder
"""
import logging

import tensorflow as tf

from src.model.builders import ModelBuilder
from src.preprocessing.utils import HEIGHT, WIDTH

RESNET_INPUT_SHAPE = (224, 224, 3)
RESNET_DENSE_UNITS = 128
RESNET_DENSE_ACTIVATION = "relu"
RESNET_DROPOUT_RATE = 0.2
RESNET_OUTPUT_ACTIVATION = "sigmoid"


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class ResNetBuilder(ModelBuilder):
    """ResNet concrete builder"""

    def set_preprocessing_layers(self):
        """Sets the preprocessing layers for the ResNet model"""
        reshape_layer = tf.keras.layers.Reshape(target_shape=(HEIGHT, WIDTH, 1))

        greyscale_layer = tf.keras.layers.Lambda(tf.image.grayscale_to_rgb)

        resize_layer = tf.keras.layers.Resizing(
            height=RESNET_INPUT_SHAPE[0], width=RESNET_INPUT_SHAPE[1]
        )

        preprocess_input = tf.keras.applications.resnet.preprocess_input

        preprocess_input_layer = tf.keras.layers.Lambda(
            preprocess_input, input_shape=RESNET_INPUT_SHAPE
        )

        self.preprocessing_layers = tf.keras.Sequential(
            [
                reshape_layer,
                greyscale_layer,
                resize_layer,
                preprocess_input_layer,
            ]
        )
        logger.info("ResNet preprocessing layers set")

    def set_model_layers(self):
        """Sets the model layers for the ResNet model"""
        base_model = tf.keras.applications.resnet.ResNet50(  
            input_shape=RESNET_INPUT_SHAPE, include_top=False, weights="imagenet"
        )

        base_model.trainable = False

        self.model_layers = base_model
        logger.info("ResNet model layers set")

    def set_output_layers(self):
        """Sets the output layers for the ResNet model"""
        pooling_layer = tf.keras.layers.GlobalAveragePooling2D()

        dense_layer = tf.keras.layers.Dense(
            units=RESNET_DENSE_UNITS, activation=RESNET_DENSE_ACTIVATION
        )

        dropout_layer = tf.keras.layers.Dropout(rate=RESNET_DROPOUT_RATE)

        output_layer = tf.keras.layers.Dense(
            1, activation=RESNET_OUTPUT_ACTIVATION
        )

        self.output_layers = tf.keras.Sequential(
            [pooling_layer, dense_layer, dropout_layer, output_layer]
        )
        logger.info("ResNet output layers set")

    def build(self):
        """Builds the ResNet model"""
        model = super().build()
        logger.info("ResNet model built")
        return model

    def reset(self):
        """Resets the ResNet builder"""
        super().reset()
        logger.info("ResNet reset")
