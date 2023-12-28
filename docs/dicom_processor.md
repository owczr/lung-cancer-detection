# Dicom Processor
## About
The `DicomProcessor` class implements the `BaseProcessor` interface. It's purpose is to process a single dicom image from the LIDC-IDRI dataset and save it in NumPy format for further usage. 

## Segmentation
Lung segmentation is used in the processing. It was implemented by combining different image processing techniques. The exact segmentation function goes like this:

1. Select threshold using OTSU algorithm
2. Create a reverse binary mask
3. Remove border
4. Remove small objects
5. Remove small holes
6. Perform binary closing
7. Perform binary opening
8. Include annotations

In some cases the part of the image with tumor could get lost in this processing and the last step ensures that it is included.
