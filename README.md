# Lung Cancer Detection

Lung Cancer Detection is a project made for Engineers Thesis "Applications of artificial intellingence in oncology on computer tomography dataset" by **Jakub Owczarek**, under the guidance of Thesis Advisor dr. hab. inz **Mariusz Mlynarczuk** prof. AGH.
<br>
The goal of this projet is to process the [LIDC-IDRI](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=1966254) dataset and fine-tune deep learining models.

## Table of Contents
- [Notebooks](#notebooks)
- [Preprocessing](#preprocessing)
- [License](#license)

## Notebooks
Notebooks were used for analysis and development of individual solutions. 

### Annotations
This notebook analyzed annotation xml file. It visualizes a nodule on image and presents a processing method.

### Diagnosis
This notebook was used to read the `tcia-diagnosis-data-2012-04-20.xls` file.

### Dicom Viewer
In this notebook we can load and see how an dicom image looks.

### Dicom
This notebook shows the various dicom tags.

### Metadata
Metadata notebook analyzed some metadata that came with dataset.

### Segmentation 
This notebook explaines each segmentation step used for creating a lung mask.

## Preprocessing
The preprocessing package contains a suite of classes and utilities to process DICOM images and XML annotations. The main classes inside this package are:

### BaseProcessor
An abstract base class that provides an interface for processing and saving data. It acts as a blueprint for all other processors.

### AnnotationProcessor
This class handles XML annotations and is responsible for processing these annotations and extracting relevant information, like the z-positions of nodules.

### DicomProcessor
This class manages DICOM files. It is designed to read DICOM files, process images, create masks, and normalize the data.

### DatasetProcessor
This class takes on entire datasets. It can process directories that contain both DICOM and XML files. With built-in parallel processing capabilities, it efficiently handles and saves processed data in an organized manner.

For a more in-depth understanding of each class, refer to their respective docstrings in the source code.

## License
This project is licensed under the MIT License - see the LICENSE.md file for details
