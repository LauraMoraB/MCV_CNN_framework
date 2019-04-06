# Object Recognition

Two networks have been created, which can be found in: ```models/networks/classification/```

- [VGG16](papers/VGG.md)
- **MiniNet**: Implemented network that stands for a convolutional neural network with low number of parameters.  


## Completed Tasks

|Task A - Run Code   | Task B - Train with KITTI  |  Task C - New Architechtures  | Task E - Documentation | 
|---|---|---|---|
| Analyse Datasets  | FineTune  | Implement MiniNet  | [Report](https://www.overleaf.com/read/jdhgqqrhcgjj) |
| Fine-Tune VGG16 for TT100K and BelgiumTSC | Train from Scratch  |   | [Slides](https://docs.google.com/presentation/d/1xWj9vOmV8CkUfDMC7wwpK70tqYfDpNb6f2E0ssXnQNs/edit?usp=sharing)|


## Results 

The following table shows the results obtained from the different tasks. We have used two networks, the well known **VGG16** and our own CNN, **MiniNet**. The experiments have been done using the following datasets: [TT100k](https://cg.cs.tsinghua.edu.cn/traffic-sign/), [BelgiumTSC](https://btsd.ethz.ch/shareddata/) and [KITTI](http://www.cvlibs.net/datasets/kitti/).

| Network |       Experiment     | TT100K  |        | Belgium |        | KITTI   |        |
|---------|----------------------|---------|--------|---------|--------|---------|--------|
|         |                      | **Val** |**Test**| **Val** |**Test**| **Val** |**Test**|
| VGG16   | Basic(ImageNet)      | 89,32   |  96.06 |  96.22  | 95.22  |  98.37  | -      |
| VGG16   | FineTune with TT100K | -       |  -     |  96.39  | 96.39  |  97.84  | -      |
| VGG16   | From scratch         | -       |  -     |   -     | -      |  97.30  | -      |
| MiniNet | From scratch         | 84.46   |  92.32 |  90.27  | 90.28  |  92.48  | -      |
