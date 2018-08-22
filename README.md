# GoogLeNet for Image Classification


- TensorFlow implementation of [Going Deeper with Convolutions](https://research.google.com/pubs/pub43022.html) (CVPR'15). 
- Architecture of GoogLeNet from the paper:
![googlenet](fig/arch.png)

## Requirements
- Python 3.3+
- [Tensorflow 1.0+](https://www.tensorflow.org/)
- [TensorCV](https://github.com/conan7882/DeepVision-tensorflow)

## Implementation Details

- The GoogLeNet model is defined in [`lib/nets/googlenet.py`](lib/nets/googlenet.py).
- Inception module is defined in [`lib/models/inception.py`](lib/models/inception.py).
- An example of image classification using pre-trained model is in [`example/pre_trained.py`](example/pre_trained.py).
- The pre-trained model on ImageNet can be downloaded [here](http://www.deeplearningmodel.net/).
- When testing the pre-trained model, images are rescaled so that the shorter dimension is 224. This is not the same as the original paper which is an ensemle of 7 similar models using 144 224x224 crops per image for testing. So the performance will not be as good as the original paper. 

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

## Usage
### Download pre-trained model
Download the pre-trained parameters [here](http://www.deeplearningmodel.net/).
### Config path
All directories are setup in [`example/setup_env.py`](example/setup_env.py).

- `PARA_DIR` is the path of the pre-trained model.
- `SAVE_DIR` is the directory to save graph summary for tensorboard. 
- `DATA_DIR` is the directory to put testing images.

### ImageNet Classification
Put test image in folder `setup_env.DATA_DIR`, then go to `example/` and run the script:

```
python pre_trained.py --type IMAGE_FILE_EXTENSION(.jpg or .png or other types of images)
```
       
   The output are the top-5 class labels and probabilities, and the top-1 human label. The structure of GoogLeNet can be viewed through TensorBoard and the summary file is saved in `setup_env.SAVE_DIR`.
   
**Note that the output label is started from 0, so label=l corresponds to the (l+1)th row in [`data/imageNetLabel.txt`](data/imageNetLabel.txt).**

