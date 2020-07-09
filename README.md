# VGG-UNet-Keras-Image-Segmentation-Customized
In this project I have used the main codes from https://github.com/divamgupta/image-segmentation-keras to generate a version to produce outputs of shapes "other than" 32x32. (currently 64x64 output).

For the current version I added a Transpose Convolution to the original code to upsample the final output utilising learning parameters. It seems using a trainable upsampling procedure is better particularly for sensitive applications such as medical imaging.


