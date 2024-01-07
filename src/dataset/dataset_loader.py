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

    def set_seed(self, seed: int) -> None:
        """Sets random seeds"""
        np.random.seed(seed)
        tf.random.set_seed(seed)

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

    def _load_batch_data(self, paths_batch, labels_batch):
        """Loads batch of data"""
        # def load(path: str) -> np.ndarray | None:
        #    try:
        #         data = np.load(path)
        #     except ValueError as e:
        #        logger.error(f"Error while loading {path}.\n{e}")
        #         return None
        #  
        # batch_data = [(load(path), label) for path, label in zip(paths_batch, labels_batch)]
        # batch_data = [(data, label) for data, label in batch_data if data is not None]
        #  
        # return batch_data
        return [(np.load(path), label) for path, label in zip(paths_batch, labels_batch)]

    def _data_generator(self):
        """Loads and yields batches of data"""
        paths, labels = self._get_data()

        paths, labels = self._shuffle(paths, labels)

        for i in range(0, len(paths), self.batch_size):
            paths_batch = paths[i : i + self.batch_size]
            labels_batch = labels[i : i + self.batch_size]
            
            try:
                batch_data = self._load_batch_data(paths_batch, labels_batch)
            except Exception as e:
                logger.error(f"Error while loading batch {i}.\n{e}")
                continue

            logger.info(f"Batch {i} loaded")

            if len(batch_data) == 0:
                continue

            data_batch, batch_labels = zip(*batch_data)

            yield np.array(data_batch), np.array(labels_batch)
