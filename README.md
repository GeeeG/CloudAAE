# CloudAAE
This is an tensorflow implementation of "CloudAAE: Learning 6D Object Pose Regression with On-line Data
Synthesis on Point Clouds"
![](figure/system_overview.png?raw=true)
# Files
1. **log**: directory to store log files during training.
2. **losses**: loss functions for training.
3. **models**: a python file defining model structure.
4. **object_model_tfrecord**: full object models for data synthesizing and visualization purpose.
5. **tf_ops**: tensorflow implementation of sampling operations (credit: Haoqiang Fan, Charles R. Qi).
6. **trained_network**: a trained network.
7. **utils**: utility files for defining model structure.
8. **ycb_video_data_tfRecords**: synthetic training data and real test data for the YCB video dataset.
9. *evaluate_cloudAAE_ycbv.py*: script for testing object 6d pose estimation with a trained network on test set in YCB video dataset.
10. *train_cloudAAE_ycbv.py*: script for training a network on synthetic data for YCB objects.


# Requirements
* Tensorflow-GPU (tested with 1.12.0)
* [transforms3d](https://matthew-brett.github.io/transforms3d/)
* [open3d](http://www.open3d.org/docs/getting_started.html) for visualization

# Test a trained network
1. Testing data in **tfrecord** format is available
* Download [zip file](https://drive.google.com/file/d/15ywcpuKVtWXzENPOaec3ZNlHPryeeiHw/view?usp=sharing)
* Unzip and place all files in **ycb_video_data_tfRecords/test_real/**
1. After activate tensorflow
```
python evaluate_cloudAAE_ycbv.py --trained_model trained_network/20200908-204328/model.ckpt --batch_size 1 --target_cls 0
```
* --trained_model: directory to trained model (*.ckpt).
* --batch_size: 1.
* --target_class: target class for pose estimation.
* Translation prediction is in unit meter.
* Rotation prediction is in axis-angle format.
3. Result
* If you turn on visualization with **b_visual=True**, you will see the following displays which are partially observed point cloud segments (red) overlaid with object model (green) with pose estimates. The reconstructed point cloud is also presented (blue).
* The coordinate is the **object coordinate**, object segment is viewed in the **camera coordinate**
<p float="center">
  <img src="/figure/0.gif" width="150" />
  <img src="/figure/14.gif" width="150" />
</p>

# Train a network
1. Training data is created syntheticly using 3D object model and 6D poses.
* The 6D pose and class id of target object are in **ycb_video_data_tfRecords/train_syn/**
2. Run script
```
python train_cloudAAE_ycbv.py
```
3. Log files and trained model is store in **log**

# Citation
If you use this code in an academic context, please consider cite the paper:

BiBTeX:
```
@inproceedings{gao2020cloudpose,
      title={CloudAAE: Learning 6D Object Pose Regression with On-line Data
Synthesis on Point Clouds},
      author={G. Gao, M. Lauri, X. Hu, J. Zhang and S. Frintrop},
      booktitle={ICRA},
      year={2021}
    }
```

# Link to Paper
TBA

# Acknowledgement
* The building block for this system is [PointNet](https://github.com/charlesq34/pointnet) and [Dynamic Graph](https://github.com/WangYueFt/dgcnn).
