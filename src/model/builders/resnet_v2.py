"""
Module with ResNetV2 concrete builder
"""
import logging

import tensorflow as tf

from src.model.builders import ModelBuilder
from src.preprocessing.utils import HEIGHT, WIDTH

RESNETV2_INPUT_SHAPE = (224, 224, 3)
RESNETV2_DENSE_UNITS = 128
RESNETV2_DENSE_ACTIVATION = "relu"
RESNETV2_DROPOUT_RATE = 0.2
RESNETV2_OUTPUT_ACTIVATION = "sigmoid"


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class ResNetV2Builder(ModelBuilder):
    """ResNetV2 concrete builder"""

    def __str__(self):
        return "ResNetV2"

    def set_preprocessing_layers(self):
        """Sets the preprocessing layers for the ResNetV2 model"""
        reshape_layer = tf.keras.layers.Reshape(target_shape=(HEIGHT, WIDTH, 1))

        greyscale_layer = tf.keras.layers.Lambda(tf.image.grayscale_to_rgb)

        resize_layer = tf.keras.layers.Resizing(
            height=RESNETV2_INPUT_SHAPE[0], width=RESNETV2_INPUT_SHAPE[1]
        )

        preprocess_input = tf.keras.applications.resnet_v2.preprocess_input

        preprocess_input_layer = tf.keras.layers.Lambda(
            preprocess_input, input_shape=RESNETV2_INPUT_SHAPE
        )

        self.preprocessing_layers = tf.keras.Sequential(
            [
                reshape_layer,
                greyscale_layer,
                resize_layer,
                preprocess_input_layer,
            ]
        )
        logger.info("ResNetV2 preprocessing layers set")

    def set_model_layers(self):
        """Sets the model layers for the ResNetV2 model"""
        base_model = tf.keras.applications.resnet_v2.ResNet50V2(  
            input_shape=RESNETV2_INPUT_SHAPE, include_top=False, weights="imagenet"
        )

        base_model.trainable = False

        self.model_layers = base_model
        logger.info("ResNetV2 model layers set")

    def set_output_layers(self):
        """Sets the output layers for the ResNetV2 model"""
        pooling_layer = tf.keras.layers.GlobalAveragePooling2D()

        dense_layer = tf.keras.layers.Dense(
            units=RESNETV2_DENSE_UNITS, activation=RESNETV2_DENSE_ACTIVATION
        )

        dropout_layer = tf.keras.layers.Dropout(rate=RESNETV2_DROPOUT_RATE)

        output_layer = tf.keras.layers.Dense(
            1, activation=RESNETV2_OUTPUT_ACTIVATION
        )

        self.output_layers = tf.keras.Sequential(
            [pooling_layer, dense_layer, dropout_layer, output_layer]
        )
        logger.info("ResNetV2 output layers set")

    def build(self):
        """Builds the ResNetV2 model"""
        model = super().build()
        logger.info("ResNetV2 model built")
        return model

    def reset(self):
        """Resets the ResNetV2 builder"""
        super().reset()
        logger.info("ResNetV2 reset")
