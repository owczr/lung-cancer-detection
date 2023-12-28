# Model Builders
## About
Model builders are classes implementing the base `ModelBuilder` interface which purpose is to build a model with:

1. Preprocessing layers
2. Base model layers
3. Output layers

## Base Models

Base Models used are taken from the [Keras Applications](https://keras.io/api/applications/). The exact models used are described in table below. For more information visit the Keras website and appropriate papers.


|Project Name|Keras Model|Parameters|Depth|
|-|-|-|-|
|ConvNeXt|ConvNeXtSmall|50.2M|-|
|DenseNet|DenseNet121|8.1M|242|
|EfficientNetV2|EfficientNetV2B0|7.2M|-|
|EfficientNet|EfficientNetB0|5.3M|132|
|InceptionResNet|InceptionResNetV2|55.9M|449|
|InceptionNet|InceptionNetV3|22.9M|189|
|MobileNet|MobileNetV3Small|2.9M|-|
|NASNet|NASNetMobile|5.3M|389|
|ResNetV2|ResNet50V2|25.6M|103|
|ResNet|ResNet50|25.6M|107|
|VGG|VGG16|138.4M|16|
|Xception|Xception|22.9M|81|
