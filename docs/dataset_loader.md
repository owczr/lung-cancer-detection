# Dataset Loader
## About
The `DatasetLoader` class was implemented to yield batches of processed dicom images from LIDC-IDRI datasets into Keras models. It implements a `get_dataset` method which returns a `tf.data.Dataset` object by using the `from_generator` method with a custom `_data_generator`. 
The `_data_generator` method loads batches of processed dicoms from `.npy` files and yields them.
