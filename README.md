# Real-Time Human Motion Capture with Multiple Depth Cameras
This is the pre-trained model of the deep convolutional network that was used in our paper:
* A. Shafaei, J. J. Little. Real-Time Human Motion Capture with Multiple Depth Cameras. In 13th Conference on Computer and Robot Vision, Victoria, Canada, 2016.

We include all the three trained models in the `models` directory. Our synthetic dataset is released separately [here](https://github.com/ashafaei/ubc3v). You can also access the project page [here](http://www.cs.ubc.ca/~shafaei/homepage/projects/crv16.php).

If you've used these models in your research, please consider citing the paper:
```bibtex
@inproceedings{Shafaei16,
  author = {Shafaei, Alireza and Little, James J.},
  title = {Real-Time Human Motion Capture with Multiple Depth Cameras},
  booktitle = {Proceedings of the 13th Conference on Computer and Robot Vision},
  year = {2016},
  organization = {Canadian Image Processing and Pattern Recognition Society (CIPPRS)},
  url = {http://www.cs.ubc.ca/~shafaei/homepage/projects/crv16.php}
}
```
If you have any questions, you can reach me at [shafaei.ca](http://shafaei.ca).

## Abstract
Commonly used human motion capture systems require intrusive attachment of markers that are visually tracked with multiple cameras. In this work we present an efficient and inexpensive solution to markerless motion capture using only a few Kinect sensors. Unlike the previous work on 3d pose estimation using a single depth camera, we relax constraints on the camera location and do not assume a co-operative user. We apply recent image segmentation techniques to depth images and use curriculum learning to train our system on purely synthetic data. Our method accurately localizes body parts without requiring an explicit shape model. The body joint locations are then recovered by combining evidence from multiple views in real-time. We also introduce a dataset of ~6 million synthetic depth frames for pose estimation from multiple cameras and exceed state-of-the-art results on the Berkeley MHAD dataset.

## Details

![alt text](http://www.cs.ubc.ca/~shafaei/homepage/projects/papers/crv_16/crv16_cnn.png "Our architecture")

Given a 250x250 depth image, this network densely classifies the pixels into the body regions of interest. `classification_demo.m` shows how the input must be pre-processed before passing it to the network.

This network is originally trained on [Caffe](https://github.com/BVLC/caffe/) but it is transfered to [MatConvnet](https://github.com/vlfeat/matconvnet) for convenience of use. It only takes 3~6 ms to classify a depth image with this architecture.

run `classification_demo.m` to run the network on the provided sample image.
![alt text](https://github.com/ashafaei/dense-depth-body-parts/raw/master/sample_gt.png "sample depth image")

### Class Reference
[!alt text](https://github.com/ashafaei/dense-depth-body-parts/raw/master/calss_ref.png "Class Reference")

### Performance
Confusion Matrix of the network trained on Hard-Pose.
[!alt text](https://github.com/ashafaei/dense-depth-body-parts/raw/master/ubc3v_confmat.png "Confusion Matrix")
