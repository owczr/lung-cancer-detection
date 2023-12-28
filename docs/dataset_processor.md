# Dataset Processor
## About
The `DatasetProcessor` implements the `BaseProcessor` interface. It's purpose is to process the whole LIDC-IDRI dataset by using `DicomProcessor` and `AnnotationProcessor` classes.

## Paralelization
Paralelization is used to speed up the processing. It was implemented by using the `ProcessPoolExecutor` from Pythons built-in `concurrent.futures` library.

## Train Test Split
After the processing is done user can split the processed directory into train and test subdirectories.

## Remove Methods
Additionally methods were implemented that can remove the processed directory or the train test split directory.
