# Semantic Segmentation

The implemented networks can be found in ```models/networks/segmentation/```:

- [FCN8](papers/FCN.md)
- [SEGNET](papers/SegNet.md)

## Completed Tasks

- [x] Task1: Analyse Datasets: Camvid, CityScapes, Kitti
- [x] Task1: Train FCN8 with Camvid
- [x] Task2: Read papers [FCN8](papers/FCN.md) and [SEGNET](papers/SEGNET.md)
- [x] Task3: Train SegNet with Camvid: using pretrained VGG16
- [x] Task4: Train FCN8 and SegNet for: Synthis-CityScapes and Kitti
- [x] Task5: Boost performance in both Networks
- [x] Task6: [Report](../README.md#Report) and Slides

## Results

The following table shows the results obtained from the different tasks. We have used two networks, the Fully Convolutional Network **FCN8** and our implementation of the **SegNet**. The experiments have been done using the following datasets: [CAamvid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/), [Synthia-CityScapes](http://synthia-dataset.net/) and [KITTI](http://www.cvlibs.net/datasets/kitti/).


| Network |  Experiment  | Camvid  |        | KITTI   |        | Synthia |        |
|---------|--------------|---------|--------|---------|--------|---------|--------|
|         |              | **Val** |**Test**| **Val** |**Test**| **Val** |**Test**|
| FCN8    | Accuracy     |   76.25 | 67.33  |  31.03  | -      | 70.05   | 70.78  |
| FCN8    | mIoU         |  65.25  | 56.81  |  26.92  | -      | 63.45   | 64.22  |
| SEGNET  | Accuracy     | 77.95   | 58.52  |  27.31  | -      | 55.30   | 55.37  |
| SEGNET  | mIoU         | 65.50   | 46.60  |  22.83  | -      | 50.00   | 49.75  |
