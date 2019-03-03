## VGG

[VGG network](https://arxiv.org/pdf/1409.1556.pdf) stands for a very deep convolutional network introduced as a solution for the ILSVR Challenge in 2014. It scored first place on the image localization task and second place on the image classification task, with an error of 7.0% in the ImageNet dataset. 

Its main contribution consists of a meticulous set of experiments with networks of increasing depth. Given the basic network architecture for image classification: a stack of convolutional layers followed by some fully connected layers and a softmax layer, it provides experiments where the configuration achievements are iteratively improved by modifying from 8-16 the number of convolutional layers.

To ensure its feasibility, filter sizes are limited to 3x3 in all layers. Hence, by using three stacked convolutional layers with 3x3 filters, instead of a convolutional layer of 7x7, they prove to decrease the number of parameters besides making the decision function more discriminative.
