from src.model.builders import XceptionBuilder
from src.model.director import ModelDirector
from src.dataset.dataset_loader import DatasetLoader 


DATASET_DIR = "LIDC-IDRI/CT/processed/train"


def build_xception():
    builder = XceptionBuilder()
    director = ModelDirector(builder)
    
    model = director.make()
    return model


def run():
    model = build_xception()
    loader = DatasetLoader(DATASET_DIR)
    dataset = loader.get_dataset()

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(dataset)


if __name__ == "__main__":
    run()
