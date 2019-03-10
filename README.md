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

* [Object Recognition](#Object-recognition)
* [Object Detection](#Object-detection)
* [Semantic Segmentation](#Semantic-segmentation)  

### Object Recognition

Two networks have been created, which can be found in: ```models/networks/classification/```

- [VGG16](papers/VGG.md)
- **MiniNet**: Implemented network that stands for a convolutional neural network with low number of parameters.  


#### Completed Tasks

|Task A - Run Code   | Task B - Train with KITTI  |  Task C - New Architechtures  | Task E - Documentation | 
|---|---|---|---|
| Analyse Datasets  | FineTune  | Implement MiniNet  | [Report](https://www.overleaf.com/read/jdhgqqrhcgjj) |
| Fine-Tune VGG16 for TT100K and BelgiumTSC | Train from Scratch  |   | [Slides](https://docs.google.com/presentation/d/1xWj9vOmV8CkUfDMC7wwpK70tqYfDpNb6f2E0ssXnQNs/edit?usp=sharing)|


#### Results 

The following table shows the results obtained from the different tasks. We have used two networks, the well known **VGG16** and our own CNN, **MiniNet**. The experiments have been done using the following datasets: [TT100k](https://cg.cs.tsinghua.edu.cn/traffic-sign/), [BelgiumTSC](https://btsd.ethz.ch/shareddata/) and [KITTI](http://www.cvlibs.net/datasets/kitti/).

| Network |       Experiment     | TT100K  |        | Belgium |        | KITTI   |        |
|---------|----------------------|---------|--------|---------|--------|---------|--------|
|         |                      | **Val** |**Test**| **Val** |**Test**| **Val** |**Test**|
| VGG16   | Basic(ImageNet)      | 89,32   |  96.06 |  96.22  | 95.22  |  98.37  | -      |
| VGG16   | FineTune with TT100K | -       |  -     |  96.39  | 96.39  |  97.84  | -      |
| VGG16   | From scratch         | -       |  -     |   -     | -      |  97.30  | -      |
| MiniNet | From scratch         | 84.46   |  92.32 |  90.27  | 90.28  |  92.48  | -      |

## Get Started

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


## Report

[Google Slide Presentation](https://docs.google.com/presentation/d/1xWj9vOmV8CkUfDMC7wwpK70tqYfDpNb6f2E0ssXnQNs/edit?usp=sharing)

[Overleaf Read-Access link](https://www.overleaf.com/read/jdhgqqrhcgjj)

## State of the Art publications

* [VGG](papers/VGG.md)
* [MobileNet](papers/MobileNet.md)

## Weights Folder
* [Link](https://drive.google.com/open?id=17LFUYLuT5L88yXYYbTEKMBGbbyaRkgea)
