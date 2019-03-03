## MobileNet

[MobileNet](https://arxiv.org/pdf/1704.04861.pdf), created by researches at Google Inc in 2017, stands for an efficient Convolutional Neural Network able to be ran on mobile phones and effective accross a wide range of applications.

Its main contribution consists on using depthwise separable convolutions -first introduced in [Xception](https://arxiv.org/pdf/1610.02357.pdf)- and introducing two new hyperparameters. Both contributions are useful for reducing the number of operations, thus providing a more effecient network. 

Depthwise separable convolutions are separated in two operations: depthwise convolutions and pointwise convolutions. The depthwise convolution maps a single convolution on each input channel obtaining the same number of output channels as in the input. Then, the pointwise convolution is applied with a 1x1 kernel to combine the features created in the previous step. Compared to the standard convolution, it results in 8 times less operations when using a 3x3 kernel.

The second contribution consists on adding two multipliers. The Width Multiplier *(alpha)* handles the trade-off between the desired latency and accuracy by thinning the number of channels from **N** to *alpha * N*, where alpha varies between 0 and 1. The Resolution Multiplier allows training MobileNet with 224x224 images and use it on 128x128 images, as it scales the images from 224 to 128. This is possible as the network uses a global average pooling instead of a flatten.