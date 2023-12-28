import os

from dotenv import load_dotenv

from src.model.builders import ...  # TODO: Import the builder
from src.model.director import ModelDirector
from src.dataset.dataset_loader import DatasetLoader 


load_dotenv()
DATASET_DIR = os.getenv("PROCESSED_DATASET_DIR") + "/train"    


# TODO: Change function name
def build_TEMPLATE():
    builder = ...  # TODO: Instantiate the builder
    director = ModelDirector(builder)
    
    model = director.make()
    return model


def run():
    model = build_TEMPLATE()
    loader = DatasetLoader(DATASET_DIR)
    dataset = loader.get_dataset()

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(dataset)


if __name__ == "__main__":
    run()
