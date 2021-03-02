import sys
import os
import tensorflow as tf
sys.path.append(os.path.join('/data_c/PointNet/pointnet/', 'tf_ops/nn_distance'))
import tf_nndistance


def get_loss(pred, label):
    """ pred: BxNx3,
        label: BxNx3, """
    dists_forward,_,dists_backward,_ = tf_nndistance.nn_distance(pred, label)
    loss_per_sample = dists_forward+dists_backward
    loss = tf.reduce_mean(loss_per_sample)
    return loss, loss_per_sample