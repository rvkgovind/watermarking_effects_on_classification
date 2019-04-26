# watermarking_effects_on_classification

VGG16 and MobileNet V1 are trained on CIFAR 10 and evaluated with watermraked images

VGG16.ipynb- trained VGGnetwork with val_accuracy of 71% tested on images taken from internet
cifar10mobilenet.ipynb-trained MobileNet with val_accuracy of 81% tested on images taken from internet
mobilenet_final.ipynb- mobilenet tested on watermarked images in Fourier domain
mobilenet_lightwatermark.ipynb- tested with spatial domain watermark with transparency of 0.1
mobilenet_heavywatermark.ipynb-tested with spatial domain watermark with transparency 0.5

The watermark image used is a cat image which is also one of the classification classes
