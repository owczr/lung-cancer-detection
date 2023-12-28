"""
Module with VGG concrete builder
"""
import logging

import tensorflow as tf

from src.model.builders import ModelBuilder
from src.preprocessing.utils import HEIGHT, WIDTH


VGG_INPUT_SHAPE = (224, 224, 3)
VGG_DENSE_UNITS = 128
VGG_DENSE_ACTIVATION = "relu"
VGG_DROPOUT_RATE = 0.2
VGG_OUTPUT_ACTIVATION = "sigmoid"


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class VGGBuilder(ModelBuilder):
    """VGG concrete builder"""

    def __str__(self):
        return "VGG"

    def set_preprocessing_layers(self):
        """Sets the preprocessing layers for the VGG model"""
        reshape_layer = tf.keras.layers.Reshape(target_shape=(HEIGHT, WIDTH, 1))

        greyscale_layer = tf.keras.layers.Lambda(tf.image.grayscale_to_rgb)

        resize_layer = tf.keras.layers.Resizing(
            height=VGG_INPUT_SHAPE[0], width=VGG_INPUT_SHAPE[1]
        )

        
        preprocess_input = tf.keras.applications.vgg16.preprocess_input

        preprocess_input_layer = tf.keras.layers.Lambda(
            preprocess_input, input_shape=VGG_INPUT_SHAPE
        )

        self.preprocessing_layers = tf.keras.Sequential(
            [
                reshape_layer,
                greyscale_layer,
                resize_layer,
                preprocess_input_layer,
            ]
        )
        logger.info("VGG preprocessing layers set")

    def set_model_layers(self):
        """Sets the model layers for the VGG model"""
        base_model = tf.keras.applications.vgg16.VGG16(  
            input_shape=VGG_INPUT_SHAPE, include_top=False, weights="imagenet"
        )

        base_model.trainable = False

        self.model_layers = base_model
        logger.info("VGG model layers set")

    def set_output_layers(self):
        """Sets the output layers for the VGG model"""
        pooling_layer = tf.keras.layers.GlobalAveragePooling2D()

        dense_layer = tf.keras.layers.Dense(
            units=VGG_DENSE_UNITS, activation=VGG_DENSE_ACTIVATION
        )

        dropout_layer = tf.keras.layers.Dropout(rate=VGG_DROPOUT_RATE)

        output_layer = tf.keras.layers.Dense(
            1, activation=VGG_OUTPUT_ACTIVATION
        )

        self.output_layers = tf.keras.Sequential(
            [pooling_layer, dense_layer, dropout_layer, output_layer]
        )
        logger.info("VGG output layers set")

    def build(self):
        """Builds the VGG model"""
        model = super().build()
        logger.info("VGG model built")
        return model

    def reset(self):
        """Resets the VGG builder"""
        super().reset()
        logger.info("VGG reset")
