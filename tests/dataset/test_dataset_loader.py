import pytest
import numpy as np
import tensorflow as tf

from src.dataset.dataset_loader import DatasetLoader

HEIGHT = 512
WIDTH = 512

BATCH_SIZE = 32

NO_IMAGES = 2

@pytest.fixture
def mock_dataset_dir(tmp_path):
    test_dir = tmp_path / "LIDC-IDRI/CT/processed/train"
    test_dir.mkdir(parents=True)

    (test_dir / "nodule").mkdir()
    (test_dir / "non_nodule").mkdir()

    np.save(test_dir / "nodule/img1.npy", np.random.rand(HEIGHT, WIDTH))
    np.save(test_dir / "non_nodule/img1.npy", np.random.rand(HEIGHT, WIDTH))
    return str(test_dir)  


class TestDatasetLoader:
    def test_initialization(self, mock_dataset_dir):
        """Test that the DatasetLoader initializes correctly."""
        loader = DatasetLoader(mock_dataset_dir, batch_size=BATCH_SIZE)
        assert loader.dataset_path == mock_dataset_dir
        assert loader.batch_size == BATCH_SIZE

    def test_get_dataset(self, mock_dataset_dir):
        """Test that the DatasetLoader returns a tf.data.Dataset."""
        loader = DatasetLoader(mock_dataset_dir)
        dataset = loader.get_dataset()
        assert isinstance(dataset, tf.data.Dataset)

    def test_dataset(self, mock_dataset_dir):
        """Test that the created tf.data.Dataset returns the correct data."""
        loader = DatasetLoader(mock_dataset_dir)
        dataset = loader.get_dataset()
        
        for x, y in dataset:
            assert x.shape == (NO_IMAGES, HEIGHT, WIDTH)
            assert y.shape == (NO_IMAGES,)
            break
