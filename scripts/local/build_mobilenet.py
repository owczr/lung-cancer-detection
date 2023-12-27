from src.model.directors import ModelDirector
from src.model.builders import MobileNetBuilder


def run():
    builder = MobileNetBuilder()
    director = ModelDirector(builder)
    model = director.make()
    print(model.summary())


if __name__ == "__main__":
    run()
