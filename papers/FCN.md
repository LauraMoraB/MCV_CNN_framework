Fully Convolutional Networks for Semantic Segmentation were proposed in 2012 by Long et al. They achieved state of the art performance in PASCAL VOC, NYUDv2 and SIFT Flow while keeping an small inference time.

The main contribution was the use of modern classification networks to tranfer the learning to the segmentation task. Given a regular classification network, the dense layers are transformed into convolutions, after these layers, a 1x1 convolution with the same number of channels as the number of classes is added. This convolution is followed by a deconvolution layer that upsamples the output of the convolution layers to the original size of the image.

It also proposes the addition of skip paths that combine the final prediction layer with lower layers. This way some detail information is passed to the last layer that otherwise would be lost because of the maxpooling layers. 
