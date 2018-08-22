# GoogLeNet for Image Classification

- TensorFlow implementation of [Going Deeper with Convolutions](https://research.google.com/pubs/pub43022.html) (CVPR'15). 
- Architecture of GoogLeNet from the paper:
![googlenet](fig/arch.png)

## Requirements
- Python 3.3+
- [TensorFlow 1.9+](https://www.tensorflow.org/)
- [Numpy](http://www.numpy.org/)
- [Scipy](https://www.scipy.org/)

## Implementation Details

- The GoogLeNet model is defined in [`src/nets/googlenet.py`](src/nets/googlenet.py).
- Inception module is defined in [`src/models/inception_module.py`](src/models/inception_module.py).
- An example of image classification using pre-trained model is in [`examples/inception_pretrained.py`](examples/inception_pretrained.py).
- When testing the pre-trained model, images are rescaled so that the shorter dimension is 224. This is not the same as the original paper which is an ensemle of 7 similar models using 144 224x224 crops per image for testing. So the performance will not be as good as the original paper. 

## Usage
### ImageNet Classification
#### Preparation
- Download the pre-trained parameters [here](https://www.dropbox.com/sh/axnbpd1oe92aoyd/AADpmuFIJTtxS7zkL_LZrROLa?dl=0). This is original from [here](http://www.deeplearningmodel.net/).
- Setup path in [`examples/inception_pretrained.py`](examples/inception_pretrained.py): `PRETRINED_PATH` is the path for pre-trained vgg model. `DATA_PATH` is the path to put testing images.

#### Run
Go to `examples/` and put test image in folder `DATA_PATH`, then run the script:

```
python inception_pretrained.py --im_name PART-OF-IMAGE-NAME
```
- `--im_name` is the option for image names you want to test. If the testing images are all `png` files, this can be `png`. The default setting is `.jpg`.
- The output will be the top-5 class labels and probabilities.


## Results
### Image classification using pre-trained model
- Top five predictions are shown. The probabilities are shown keeping two decimal places. Note that the pre-trained model are trained on [ImageNet](http://www.image-net.org/).
- result of VGG19 for the same images can be found [here](https://github.com/conan7882/VGG-tensorflow#results). 
**The pre-processing of images for both experiments are the same.** 

*Data Source* | *Image* | *Result* |
|:--|:--:|:--|
[COCO](http://cocodataset.org/#home) |<img src='data/000000000285.jpg' height='200px'>| 1: probability: 1.00, label: brown bear, bruin, Ursus arctos<br>2: probability: 0.00, label: ice bear, polar bear<br>3: probability: 0.00, label: hyena, hyaena<br>4: probability: 0.00, label: chow, chow chow<br>5: probability: 0.00, label: American black bear, black bear
[COCO](http://cocodataset.org/#home) |<img src='data/000000000724.jpg' height='200px'>| 1: probability: 0.79, label: street sign<br>2: probability: 0.06, label: traffic light, traffic signal, stoplight<br>3: probability: 0.03, label: parking meter<br>4: probability: 0.02, label: mailbox, letter box<br>5: probability: 0.01, label: balloon
[COCO](http://cocodataset.org/#home) |<img src='data/000000001584.jpg' height='200px'>|1: probability: 0.94, label: trolleybus, trolley coach<br>2: probability: 0.05, label: passenger car, coach, carriage<br>3: probability: 0.00, label: fire engine, fire truck<br>4: probability: 0.00, label: streetcar, tram, tramcar, trolley<br>5: probability: 0.00, label: minibus
[COCO](http://cocodataset.org/#home) |<img src='data/000000003845.jpg' height='200px'>|1: probability: 0.35, label: burrito<br>2: probability: 0.17, label: potpie<br>3: probability: 0.14, label: mashed potato<br>4: probability: 0.10, label: plate<br>5: probability: 0.03, label: pizza, pizza pie
[ImageNet](http://www.image-net.org/) |<img src='data/ILSVRC2017_test_00000004.jpg' height='200px'>|1: probability: 1.00, label: goldfish, Carassius auratus<br>2: probability: 0.00, label: rock beauty, Holocanthus tricolor<br>3: probability: 0.00, label: puffer, pufferfish, blowfish, globefish<br>4: probability: 0.00, label: tench, Tinca tinca<br>5: probability: 0.00, label: anemone fish
Self Collection | <img src='data/IMG_4379.jpg' height='200px'>|1: probability: 0.32, label: Egyptian cat<br>2: probability: 0.30, label: tabby, tabby cat<br>3: probability: 0.05, label: tiger cat<br>4: probability: 0.02, label: mouse, computer mouse<br>5: probability: 0.02, label: paper towel
Self Collection | <img src='data/IMG_7940.JPG' height='200px'>|1: probability: 1.00, label: streetcar, tram, tramcar, trolley, trolley car<br>2: probability: 0.00, label: passenger car, coach, carriage<br>3: probability: 0.00, label: trolleybus, trolley coach, trackless trolley<br>4: probability: 0.00, label: electric locomotive<br>5: probability: 0.00, label: freight car

## Author
Qian Ge
