import os

from dotenv import load_dotenv

from src.model.builders import InceptionNetBuilder
from src.model.director import ModelDirector
from src.dataset.dataset_loader import DatasetLoader 


load_dotenv()
DATASET_DIR = os.getenv("PROCESSED_DATASET_DIR") + "/train"


def build_inceptionnet():
    builder = InceptionNetBuilder()
    director = ModelDirector(builder)
    
    model = director.make()
    return model


def run():
    model = build_inceptionnet()
    loader = DatasetLoader(DATASET_DIR)
    dataset = loader.get_dataset()

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(dataset)


if __name__ == "__main__":
    run()
