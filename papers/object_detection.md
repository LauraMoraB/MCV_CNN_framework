# Semantic Segmentation

A new framework using faster-RCNN network has been integrated, which can be found in: ```object_detection/faster_rcnn.pytorch/```

- [FasterRCNN](papers/FasterRCNN.md)

## Completed Tasks

- [x] Task1: Train an existing object detection network
- [x] Task2: Read papers [FasterRCNN](papers/FasterRCNN.md) and [YOLO](YOLO.md)
- [x] Task3: Train the network for another dataset: Pascal and Udacity
- [x] Task4: Boost performance in both Networks
- [x] Task5: [Report](../README.md#Report) and Slides

## Results

The following table shows the results obtained from the different tasks. We have used Faster RCNN network and the experiments have been done using the following datasets: [Pascal](https://link.springer.com/article/10.1007/s11263-009-0275-4) and 
[Udacity](https://github.com/udacity/self-driving-car/tree/master/datasets).


| Network |  Metric  | Pascal VOC first approach  |        | Pascal VOC Best   |        | Udacity |        |
|---------|--------------|---------|--------|---------|--------|---------|--------|
|         |              | **Val** |**Test**| **Val** |**Test**| **Val** |**Test**|
| FasterRCNN    | mAP     |   0.66 | 0.60  |  0.92  | 0.73      | 0.81   | 0.44  |
