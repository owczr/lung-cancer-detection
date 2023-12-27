from src.model.builders import MobileNetBuilder
from src.model.director import ModelDirector


def build_mobilenet():
    builder = MobileNetBuilder()
    director = ModelDirector(builder)
    model = director.make()
    return model


def run():
    model = build_mobilenet()


if __name__ == "__main__":
    run()
