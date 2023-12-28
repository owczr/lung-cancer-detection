import os

from dotenv import load_dotenv

from src.model.builders import EfficientNetBuilder
from src.model.director import ModelDirector
from src.dataset.dataset_loader import DatasetLoader

load_dotenv()
DATASET_DIR = os.getenv("PROCESSED_DATASET_DIR") + "/train"


def build_efficientnet():
    builder = EfficientNetBuilder()
    director = ModelDirector(builder)

    model = director.make()
    return model


def run():
    model = build_efficientnet()
    loader = DatasetLoader(DATASET_DIR)
    dataset = loader.get_dataset()

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(dataset)


if __name__ == "__main__":
    run()
