# M5 Project: Scene Understanding for Autonomous Vehicles

The goal of this project is to learn the basic concepts and techniques to build deep neural networks to detect, segment and recognize specific objects, focusing on the self-driving car application. With the aim to solve the problem of automatic image understanding, the tasks performed include object recognition, detection and semantic segmentation in images recorded by an on-board vehicle camera.

## Team members

* Daniel Azemar ([daniel.azemar@e-campus.uab.cat](mailto:daniel.azemar@e-campus.uab.cat))
* María Gil Aragones ([maria.gilaragones@gmail.com](mailto:maria.gilaragones@gmail.com))
* Laura Mora Ballestar ([lmoraballestar@gmail.com](mailto:lmoraballestar@gmail.com))
* Richard Segovia ([richard.segovia@e-campus.uab.cat](mailto:richard.segovia@e-campus.uab.cat))

## Index

* [Applications](#Applications)
* [Get Started](#Get-Started)
* [Report](#Report)
* [State of the Art](#State-of-the-art-publications)

## Applications

This repository creates a PyTorch based framework to achieve three goals:

* [Object Recognition](papers/object_recognition.md)
* [Object Detection](papers/object_detection.md)
* [Semantic Segmentation](papers/semantic_segmentation.md)  


## Get Started

### **Object Recognition and Semantic Segmentation**
### Installation

Environment Set Up:

* Python 3.7
* Pytorch -- cudatoolkit, torchvision

```bash
pip install -r requirements.txt
```

### Run the code

```bash
# --exp_name: directory where results are stored
# --config_dile: file where the configuration for code is set up
python3 main.py --exp_name dir_name --exp_folder ./ --config_file config/configFile.yml
```
### **Object Detection**

### Installation

In order to execute the framework for object detection, different steps have to be followed. First, see source [repository](https://github.com/jwyang/faster-rcnn.pytorch/tree/pytorch-1.0)

**1. Prerequisits**
- Python 3.6
- Pytorch 1.0
- Cuda 8 or hihger

**2. Data preparation**

The framework requires COCO and PASCAL to be installed in order to work properly

* **PASCAL_VOC 07+12**: Please follow the instructions in [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models) to prepare VOC datasets. After downloading the data, create softlinks in the folder `object_detection/faster-rcnn.pytorch/data/`.

* **COCO**: Download from the respository [COCOAPI](https://github.com/pdollar/coco) and store in folder `object_detection/faster-rcnn.pytorch/data/`

* **UDACITY**

    **TODO** -- podem posar el que hem necessitat fer per incorporar-lo

**3. Pretrained Models**

The framework uses VGG16 or Restnet101 as baseline architectures. The weights of the networks, trained with Caffe, must be stored in the folder `object_detection/framework/pretrained_models/`

Link to download the models from the source repository:

* VGG16: [Dropbox](https://www.dropbox.com/s/s3brpk0bdq60nyb/vgg16_caffe.pth?dl=0)

* ResNet101: [Dropbox](https://www.dropbox.com/s/iev3tkbz5wyyuz9/resnet101_caffe.pth?dl=0)

**4. Compilation**
```bash
pip install -r requirements.txt

cd lib
python setup.py build develop
```
### Run the code

**Train**

```bash
LEARNING_RATE=lr
BATCH_SIZE=batchSize
DECAY_STEP=decayStep
DATASET=udacity_voc #udacity_voc or pascal_voc
NETWORK=res101 #res101 or vgg16 
EPOCHS=numberEpochs

python3 trainval_net.py --dataset $DATASET --net $NETWORK \
                       --bs $BATCH_SIZE --nw 1 \
                       --lr $LEARNING_RATE --lr_decay_step $DECAY_STEP \
                       --cuda --mGPUs --epochs $EPOCHS

```

**Test**
```bash
python3 test_net.py --dataset  $DATASET --net $NETWORK \
                       --cuda --mGPUs --checksession $CHECK_SESSION --checkepoch $CHECK_EPOCH --checkpoint $CHECK_MODEL
```

**Demo**

Script which loads the trained model and saves the result image detection in the folder `object_detection/framework/images/`

```bash
python demo.py --net res101 \
               --checksession $SESSION --checkepoch $EPOCH --checkpoint $CHECKPOINT --cuda --load_dir models/

```

## Report

|Object Recognition | Semantic Segmentation | Object Detection |
|-------------------| ----------------------| -----------------|
|[Presentation](https://docs.google.com/presentation/d/1xWj9vOmV8CkUfDMC7wwpK70tqYfDpNb6f2E0ssXnQNs/edit?usp=sharing)|[Presentation](https://docs.google.com/presentation/d/1FM0sqHXvJMrfRbRdjOkyXKsVXi6Fi8xuY1Yegxxsydo/edit?usp=sharing)|[Presentation](https://docs.google.com/presentation/d/1KHaQK2LPUhY63ut-xJkDmVtclmcCocw21XxlizU6IA4/edit?usp=sharing)|


### Complete Report
[Overleaf Read-Access link](https://www.overleaf.com/read/jdhgqqrhcgjj)

## State of the Art publications

* [VGG](papers/VGG.md)
* [MobileNet](papers/MobileNet.md)
* [FCN](papers/FCN.md)
* [SegNet](papers/SegNet.md)
* [FRCNN](papers/FasterRCNN.md)
* [YOLO](papers/YOLO.md)

## Weights Folder
* [Link](https://drive.google.com/open?id=17LFUYLuT5L88yXYYbTEKMBGbbyaRkgea)
