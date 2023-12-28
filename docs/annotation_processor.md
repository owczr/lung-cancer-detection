# Annotation Processor
## About
`AnnotationProcessor` class implements the `BaseProcessor` interface. It's purpose is to process the XML annotation files provided by the LIDC-IDRI dataset into a format that will be used for image classification.

## Format
The format used for image classification is just the z position of the slice. It indicates whether a tumor is present on the image slice or not.