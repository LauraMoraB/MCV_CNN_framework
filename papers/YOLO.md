# YOLO

"You Only Look Once" [YOLO](https://arxiv.org/pdf/1506.02640v5.pdf) architecture can be defined, in contrast to fast RCNN, as a fully convolutional neural netowrk, where the input image (nxn) is fed to the network and the output is the predition (mxm). Basically, the architecture splits the image in a mxm grid generating bounding boxes and class probabilities for each box. The architecture of the network has 24 Conv.Layers and 2 FC.

Two interesting changes to the loss funciton are the differential weights for predictions in boxes containg an object or not during training and predict the square error to penalize error in small objects or large objects differently.

Currently, [YoloV2]( https://arxiv.org/pdf/1612.08242v1.pdf) can read up to 45 frames/second and the network can understand generalized object representations, so it can be trained in real images and artwork. For [YoloV3](https://pjreddie.com/media/files/papers/YOLOv3.pdf), the perfomance has been increased thanks to multi-scale predictions and using binary cross-entropy loss for each label, which improves class prediction as well as reduction computation complexity by avoiding the softmax function. 