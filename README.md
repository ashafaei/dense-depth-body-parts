# Real-Time Human Motion Capture with Multiple Depth Cameras
This is the pre-trained model of the deep convolutional network that was used in our paper:
* A. Shafaei, J. J. Little. Real-Time Human Motion Capture with Multiple Depth Cameras. In 13th Conference on Computer and Robot Vision, Victoria, Canada, 2016.

## Abstract
Commonly used human motion capture systems require intrusive attachment of markers that are visually tracked with multiple cameras. In this work we present an efficient and inexpensive solution to markerless motion capture using only a few Kinect sensors. Unlike the previous work on 3d pose estimation using a single depth camera, we relax constraints on the camera location and do not assume a co-operative user. We apply recent image segmentation techniques to depth images and use curriculum learning to train our system on purely synthetic data. Our method accurately localizes body parts without requiring an explicit shape model. The body joint locations are then recovered by combining evidence from multiple views in real-time. We also introduce a dataset of ~6 million synthetic depth frames for pose estimation from multiple cameras and exceed state-of-the-art results on the Berkeley MHAD dataset.

## Details

![alt text](http://www.cs.ubc.ca/~shafaei/homepage/projects/papers/crv_16/crv16_cnn.png "Our architecture")

This network is originally trained on [Caffe](https://github.com/BVLC/caffe/) but it is transfered to [MatConvnet](https://github.com/vlfeat/matconvnet) for convenience of use. It only takes 3~6 ms to classify a depth image with this architecture.

