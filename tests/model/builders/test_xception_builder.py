import pytest
import tensorflow as tf

from src.model.builders import XceptionBuilder
from src.model.director import ModelDirector


@pytest.fixture
def builder():
    return XceptionBuilder()


@pytest.fixture
def director():
    return ModelDirector(builder=XceptionBuilder())


class TestTemplateBuilder:  # TODO: Change the class name
    def test_set_preprocessing_layers(self, builder):
        builder.set_preprocessing_layers()
        assert isinstance(
            builder.preprocessing_layers, tf.keras.Sequential
        ), "Preprocessing layers should be a Keras Sequential model."

    def test_set_model_layers(self, builder):
        builder.set_model_layers()
        assert isinstance(
            builder.model_layers, tf.keras.Model
        ), "Model layers should be a Keras Model instance."

    def test_set_output_layers(self, builder):
        builder.set_output_layers()
        assert isinstance(
            builder.output_layers, tf.keras.Sequential
        ), "Output layers should be a Keras Sequential model."

    def test_build(self, builder):
        builder.set_preprocessing_layers()
        builder.set_model_layers()
        builder.set_output_layers()
        model = builder.build()
        assert isinstance(
            model, tf.keras.Model
        ), "Build should return a Keras Model instance."

    def test_reset(self, builder):
        builder.set_preprocessing_layers()
        builder.set_model_layers()
        builder.set_output_layers()
        builder.reset()
        assert builder.model is None, "Model should be None after reset."

    def test_model_director(self, director):
        model = director.make()
        assert isinstance(
            model, tf.keras.Model
        ), "Model director should return a Keras Model instance."
