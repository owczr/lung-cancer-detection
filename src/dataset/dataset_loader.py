import os
import logging

import numpy as np
import tensorflow as tf

from src.preprocessing.utils import NODULE, NON_NODULE, HEIGHT, WIDTH


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DatasetLoader:
    """LIDC-IDRI dataset loader"""

    def __init__(self, dataset_path, batch_size=32):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self._dataset = tf.data.Dataset
        logger.info(
            f"Initialized DatasetLoader with dataset path: {dataset_path} and batch size: {batch_size}"
        )

    def get_dataset(self) -> tf.data.Dataset:
        """Returns tf.data.Dataset"""
        return self._dataset.from_generator(
            self._data_generator,
            output_types=(tf.float64, tf.uint8),
            output_shapes=(
                tf.TensorShape([None, HEIGHT, WIDTH]),  # None for partial batch size
                tf.TensorShape([None]),
            ),
        )

    def _get_data(self):
        """Returns flat list of paths to nodule and non-nodule images with labels"""
        nodule_path = os.path.join(self.dataset_path, NODULE)
        non_nodule_path = os.path.join(self.dataset_path, NON_NODULE)

        nodule_paths = [
            os.path.abspath(os.path.join(nodule_path, path))
            for path in os.listdir(nodule_path)
        ]
        non_nodule_paths = [
            os.path.abspath(os.path.join(non_nodule_path, path))
            for path in os.listdir(non_nodule_path)
        ]

        nodule_labels = [1] * len(nodule_paths)
        non_nodule_labels = [0] * len(non_nodule_paths)

        paths = nodule_paths + non_nodule_paths
        labels = nodule_labels + non_nodule_labels

        logger.info("Dataset loaded")

        return paths, labels

    def _shuffle(self, paths, labels):
        """Shuffles dataset"""
        paths, labels = np.array(paths), np.array(labels)

        indices = np.arange(len(paths))
        np.random.shuffle(indices)

        paths, labels = paths[indices], labels[indices]

        logger.info("Dataset shuffled")

        return paths, labels

    def _load_batch_data(self, paths_batch):
        """Loads batch of data"""
        batch_data = [np.load(path) for path in paths_batch]
        return batch_data

    def _data_generator(self):
        """Loads and yields batches of data"""
        paths, labels = self._get_data()

        paths, labels = self._shuffle(paths, labels)

        for i in range(0, len(paths), self.batch_size):
            paths_batch = paths[i : i + self.batch_size]
            labels_batch = labels[i : i + self.batch_size]

            batch_data = self._load_batch_data(paths_batch)

            logger.info(f"Batch {i} loaded")

            yield np.array(batch_data), np.array(labels_batch)
