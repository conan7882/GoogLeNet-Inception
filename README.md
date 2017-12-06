# GoogLeNet for Image Classification


- TensorFlow implementation of [Going Deeper with Convolutions](https://research.google.com/pubs/pub43022.html) (CVPR'15). 


## Requirements
- Python 3.3+
- [Tensorflow 1.0+](https://www.tensorflow.org/)
- [TensorCV](https://github.com/conan7882/DeepVision-tensorflow)

## Implementation Details

- The GoogLeNet model is defined in [`googlenet.py`](lib/models/googlenet.py).
- Inception module is defined in [`inception.py`](lib/models/inception.py).
- An example of image classification using pre-trained model is in [`pre_trained.py`](example/pre_trained.py).
- The pre-trained model on ImageNet can be downloaded [here](http://www.deeplearningmodel.net/).
- When testing the pre-trained model, images are rescaled so that the shorter dimension is 224. Then the center 224x224 crop is used as the input of the network. This is not the same as the original paper which use 144 crops per image for testing. So the performance will not be as good as the original paper. However, it is easy to add a pre-processing step to obtain the 144 crops and test the classification performance.
- Since the there is a [Global average pooling](https://arxiv.org/abs/1312.4400) layer before fully connected layer, the input image can be arbitrary size. But the performance is worse when I feed the network using the original image size.

## Results
- ### Image classification on ImageNet
<div align='left'>
  <img src='fig/1.png' height='300px'>
  <img src='fig/2.png' height="300px">
  <img src='fig/3.png' height="300px">
</div>
- ### Images from my photo collection
<div align='left'>
  <img src='fig/4.png' height='350px'>
  <img src='fig/5.png' height="350px">
</div>


