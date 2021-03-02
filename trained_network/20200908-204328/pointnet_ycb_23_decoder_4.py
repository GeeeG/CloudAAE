""" TF model for point cloud autoencoder. PointNet encoder, FC decoder.
Using GPU Chamfer's distance loss.

Author: Charles R. Qi
Date: May 2018
"""
import tensorflow as tf
import numpy as np
import math
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import tf_util
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/nn_distance'))
import tf_nndistance
from pointnet_util_late_class import pointnet_sa_module

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    return pointclouds_pl, labels_pl


def get_model_pn(point_cloud, is_training, bn_decay=None):
    """ Autoencoder for point clouds.
    Input:
        point_cloud: TF tensor BxNx3
        is_training: boolean
        bn_decay: float between 0 and 1
    Output:
        net: TF tensor BxNx3, reconstructed point clouds
        end_points: dict
    """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    point_dim = point_cloud.get_shape()[2].value
    end_points = {}
    input_image = tf.expand_dims(point_cloud, -1)
    # Encoder
    net = tf_util.conv2d(input_image, 64, [1,point_dim],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv1_decoder', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv2_decoder', bn_decay=bn_decay)

    point_feat = net

    net = tf_util.conv2d(point_feat, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv3_decoder', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv4_decoder', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 1024, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv5_decoder', bn_decay=bn_decay)
    global_feat = tf_util.max_pool2d(net, [num_point,1],
                                     padding='VALID', scope='maxpool_decoder')

    global_feat_expand = tf.tile(global_feat, [1, num_point, 1, 1])
    concat_feat = tf.concat([point_feat, global_feat_expand], 3)

    # FC Decoder
    # net, _, _ = tf_util.fully_connected(net, 1024, bn=True, is_training=is_training, scope='fc1_decoder', bn_decay=bn_decay)
    # net, _, _ = tf_util.fully_connected(net, 1024, bn=True, is_training=is_training, scope='fc2_decoder', bn_decay=bn_decay)
    # net, _, _ = tf_util.fully_connected(net, num_point*3, activation_fn=None, scope='fc3_decoder')
    net = tf_util.conv2d(concat_feat, 512, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv6_decoder', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 256, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv7_decoder', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv8_decoder', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv9_decoder', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 3*4, [1, 1],
                         padding='VALID', stride=[1, 1], activation_fn=None,
                         scope='conv10_decoder')
    net = tf.reshape(net, (batch_size, num_point*4, 3))

    return net, end_points


# pointnet ++
# takes too much memory
def get_model_pnpp(point_cloud, cls_gt_onehot, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    feature_dim = point_cloud.get_shape()[2].value
    end_points = {}
    l0_xyz = point_cloud
    l0_points = None
    end_points['l0_xyz'] = l0_xyz

    # Set abstraction layers
    # Note: When using NCHW for layer 2, we see increased GPU memory usage (in TF1.4).
    # So we only use NCHW for layer 1 until this issue can be resolved.
    l1_xyz, l1_points, l1_indices, l1_group_xyz = pointnet_sa_module(l0_xyz, l0_points, npoint=128, radius=0.2, nsample=32,
                                                       mlp=[64, 64, 128], mlp2=None, group_all=False,
                                                       is_training=is_training, bn_decay=bn_decay, scope='layer1_pnpp',
                                                       use_nchw=True)
    l2_xyz, l2_points, l2_indices, l2_group_xyz = pointnet_sa_module(l1_xyz, l1_points, npoint=64, radius=0.4, nsample=64,
                                                       mlp=[128, 128, 256], mlp2=None, group_all=False,
                                                       is_training=is_training, bn_decay=bn_decay, scope='layer2_pnpp')
    l3_xyz, l3_points, l3_indices, l3_group_xyz = pointnet_sa_module(l2_xyz, l2_points, npoint=None, radius=None, nsample=None,
                                                       mlp=[256, 512, 1024], mlp2=None, group_all=True,
                                                       is_training=is_training, bn_decay=bn_decay, scope='layer3_pnpp')

    # Fully connected layers
    net = tf.reshape(l3_points, [batch_size, -1])
    net = tf.concat([net, cls_gt_onehot], axis=1)
    net, _, _ = tf_util.fully_connected(net, 1024, bn=True, is_training=is_training, scope='fc1_pnpp', bn_decay=bn_decay)
    net, _, _ = tf_util.fully_connected(net, 1024, bn=True, is_training=is_training, scope='fc2_pnpp', bn_decay=bn_decay)
    net, _, _ = tf_util.fully_connected(net, num_point * 3 * 4, activation_fn=None, scope='fc3_pnpp')

    net = tf.reshape(net, (batch_size, num_point * 4, 3))


    return net, end_points


# DGCNN
def get_model_dgcnn(point_cloud, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    input_image = tf.expand_dims(point_cloud, -1)

    print "###########point_cloud", point_cloud

    k = 10
    adj_matrix = tf_util.pairwise_xyz_distance(point_cloud)
    nn_idx = tf_util.knn(adj_matrix, k=k)
    edge_feature = tf_util.get_edge_feature(point_cloud, nn_idx=nn_idx, k=k)

    print "##########edge_feature", edge_feature

    net = tf_util.conv2d(edge_feature, 64, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='dgcnn1', bn_decay=bn_decay)

    print "##########dgcnn1", net

    net = tf.reduce_max(net, axis=-2, keep_dims=True)
    net1 = net

    print "#########reduce_max net1", net1

    adj_matrix = tf_util.pairwise_xyz_distance(net)
    nn_idx = tf_util.knn(adj_matrix, k=k)
    edge_feature = tf_util.get_edge_feature(net, nn_idx=nn_idx, k=k)

    print "###########edge feature", edge_feature

    net = tf_util.conv2d(edge_feature, 64, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='dgcnn2', bn_decay=bn_decay)

    print "###########dgcnn2", net

    net = tf.reduce_max(net, axis=-2, keep_dims=True)
    net2 = net

    print "###########reduce_max net2", net2

    adj_matrix = tf_util.pairwise_xyz_distance(net)
    nn_idx = tf_util.knn(adj_matrix, k=k)
    edge_feature = tf_util.get_edge_feature(net, nn_idx=nn_idx, k=k)

    print "###########edge feature", edge_feature

    net = tf_util.conv2d(edge_feature, 64, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='dgcnn3', bn_decay=bn_decay)

    print "###########dgcnn3", net

    net = tf.reduce_max(net, axis=-2, keep_dims=True)
    net3 = net

    print "###########reduce_max net3", net3

    adj_matrix = tf_util.pairwise_xyz_distance(net)
    nn_idx = tf_util.knn(adj_matrix, k=k)
    edge_feature = tf_util.get_edge_feature(net, nn_idx=nn_idx, k=k)

    print "###########edge feature", edge_feature

    net = tf_util.conv2d(edge_feature, 128, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='dgcnn4', bn_decay=bn_decay)

    print "###########dgcnn4", net

    net = tf.reduce_max(net, axis=-2, keep_dims=True)
    net4 = net

    print "###########reduce_max net4", net4

    net = tf_util.conv2d(tf.concat([net1, net2, net3, net4], axis=-1), 1024, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='dgcnn_agg', bn_decay=bn_decay)

    max_indices = tf.argmax(net, axis=1)

    print "############agg", net

    net = tf.reduce_max(net, axis=1, keep_dims=True)

    print "###########reduce_max", net


    # MLP on global point cloud vector
    net = tf.reshape(net, [batch_size, -1])
    net, _, _ = tf_util.fully_connected(net, 1024, bn=True, is_training=is_training,
                                  scope='dgcnn_fc1', bn_decay=bn_decay)
    print "##########fc1", net

    net, _, _ = tf_util.fully_connected(net, 1024, bn=True, is_training=is_training,
                                  scope='dgcnn_fc2', bn_decay=bn_decay)

    print "##########fc2", net
    # net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
    #                       scope='dp1')
    # net = tf.contrib.layers.unit_norm(net, dim=1)  # to have unit activation to last layer

    net, out_weight, out_biases = tf_util.fully_connected(net, num_point*3*4, activation_fn=None, scope='dgcnn_output')

    net = tf.reshape(net, (batch_size, num_point * 4, 3))

    return net, end_points


def get_model_dgcnn_mean(point_cloud, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    input_image = tf.expand_dims(point_cloud, -1)

    print "###########point_cloud", point_cloud

    k = 10
    adj_matrix = tf_util.pairwise_xyz_distance(point_cloud)
    nn_idx = tf_util.knn(adj_matrix, k=k)
    edge_feature = tf_util.get_edge_feature(point_cloud, nn_idx=nn_idx, k=k)

    print "##########edge_feature", edge_feature

    net = tf_util.conv2d(edge_feature, 64, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='dgcnn1', bn_decay=bn_decay)

    print "##########dgcnn1", net

    net = tf.reduce_mean(net, axis=-2, keep_dims=True)
    net1 = net

    print "#########reduce_mean net1", net1

    adj_matrix = tf_util.pairwise_xyz_distance(net)
    nn_idx = tf_util.knn(adj_matrix, k=k)
    edge_feature = tf_util.get_edge_feature(net, nn_idx=nn_idx, k=k)

    print "###########edge feature", edge_feature

    net = tf_util.conv2d(edge_feature, 64, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='dgcnn2', bn_decay=bn_decay)

    print "###########dgcnn2", net

    net = tf.reduce_mean(net, axis=-2, keep_dims=True)
    net2 = net

    print "###########reduce_mean net2", net2

    adj_matrix = tf_util.pairwise_xyz_distance(net)
    nn_idx = tf_util.knn(adj_matrix, k=k)
    edge_feature = tf_util.get_edge_feature(net, nn_idx=nn_idx, k=k)

    print "###########edge feature", edge_feature

    net = tf_util.conv2d(edge_feature, 64, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='dgcnn3', bn_decay=bn_decay)

    print "###########dgcnn3", net

    net = tf.reduce_mean(net, axis=-2, keep_dims=True)
    net3 = net

    print "###########reduce_mean net3", net3

    adj_matrix = tf_util.pairwise_xyz_distance(net)
    nn_idx = tf_util.knn(adj_matrix, k=k)
    edge_feature = tf_util.get_edge_feature(net, nn_idx=nn_idx, k=k)

    print "###########edge feature", edge_feature

    net = tf_util.conv2d(edge_feature, 128, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='dgcnn4', bn_decay=bn_decay)

    print "###########dgcnn4", net

    net = tf.reduce_mean(net, axis=-2, keep_dims=True)
    net4 = net

    print "###########reduce_mean net4", net4

    net = tf_util.conv2d(tf.concat([net1, net2, net3, net4], axis=-1), 1024, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='dgcnn_agg', bn_decay=bn_decay)

    max_indices = tf.argmax(net, axis=1)

    print "############agg", net

    net = tf.reduce_mean(net, axis=1, keep_dims=True)

    print "###########reduce_mean", net


    # MLP on global point cloud vector
    net = tf.reshape(net, [batch_size, -1])
    end_points['embedding'] = net
    print "###########bottle neck", net

    net, _, _ = tf_util.fully_connected(net, 1024, bn=True, is_training=is_training,
                                  scope='dgcnn_fc1', bn_decay=bn_decay)
    print "##########fc1", net

    net, _, _ = tf_util.fully_connected(net, 1024, bn=True, is_training=is_training,
                                  scope='dgcnn_fc2', bn_decay=bn_decay)

    print "##########fc2", net
    # net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
    #                       scope='dp1')
    # net = tf.contrib.layers.unit_norm(net, dim=1)  # to have unit activation to last layer

    net, out_weight, out_biases = tf_util.fully_connected(net, num_point*3*4, activation_fn=None, scope='dgcnn_output')

    net = tf.reshape(net, (batch_size, num_point * 4, 3))

    return net, end_points


def get_model_dgcnn_mean_6d(point_cloud, is_training_pl_encoder, is_training, k_neighbor, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    input_image = tf.expand_dims(point_cloud, -1)

    print "###########point_cloud", point_cloud

    k = k_neighbor
    adj_matrix = tf_util.pairwise_xyz_distance(point_cloud)
    nn_idx = tf_util.knn(adj_matrix, k=k)
    edge_feature = tf_util.get_edge_feature(point_cloud, nn_idx=nn_idx, k=k)

    print "##########edge_feature", edge_feature

    net = tf_util.conv2d(edge_feature, 64, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training_pl_encoder,
                         scope='dgcnn1', bn_decay=bn_decay)

    print "##########dgcnn1", net

    net = tf.reduce_mean(net, axis=-2, keep_dims=True)
    net1 = net

    print "#########reduce_mean net1", net1

    adj_matrix = tf_util.pairwise_xyz_distance(net)
    nn_idx = tf_util.knn(adj_matrix, k=k)
    edge_feature = tf_util.get_edge_feature(net, nn_idx=nn_idx, k=k)

    print "###########edge feature", edge_feature

    net = tf_util.conv2d(edge_feature, 64, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training_pl_encoder,
                         scope='dgcnn2', bn_decay=bn_decay)

    print "###########dgcnn2", net

    net = tf.reduce_mean(net, axis=-2, keep_dims=True)
    net2 = net

    print "###########reduce_mean net2", net2

    adj_matrix = tf_util.pairwise_xyz_distance(net)
    nn_idx = tf_util.knn(adj_matrix, k=k)
    edge_feature = tf_util.get_edge_feature(net, nn_idx=nn_idx, k=k)

    print "###########edge feature", edge_feature

    net = tf_util.conv2d(edge_feature, 64, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training_pl_encoder,
                         scope='dgcnn3', bn_decay=bn_decay)

    print "###########dgcnn3", net

    net = tf.reduce_mean(net, axis=-2, keep_dims=True)
    net3 = net

    print "###########reduce_mean net3", net3

    adj_matrix = tf_util.pairwise_xyz_distance(net)
    nn_idx = tf_util.knn(adj_matrix, k=k)
    edge_feature = tf_util.get_edge_feature(net, nn_idx=nn_idx, k=k)

    print "###########edge feature", edge_feature

    net = tf_util.conv2d(edge_feature, 128, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training_pl_encoder,
                         scope='dgcnn4', bn_decay=bn_decay)

    print "###########dgcnn4", net

    net = tf.reduce_mean(net, axis=-2, keep_dims=True)
    net4 = net

    print "###########reduce_mean net4", net4

    net = tf_util.conv2d(tf.concat([net1, net2, net3, net4], axis=-1), 1024, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training_pl_encoder,
                         scope='dgcnn_agg', bn_decay=bn_decay)

    max_indices = tf.argmax(net, axis=1)

    print "############agg", net

    net = tf.reduce_mean(net, axis=1, keep_dims=True)

    print "###########reduce_mean", net


    # MLP on global point cloud vector
    net = tf.reshape(net, [batch_size, -1])
    end_points['embedding'] = net
    print "###########bottle neck", net

    net, _, _ = tf_util.fully_connected(net, 1024, bn=True, is_training=is_training,
                                  scope='dgcnn_fc1', bn_decay=bn_decay)
    print "##########fc1", net

    net, _, _ = tf_util.fully_connected(net, 1024, bn=True, is_training=is_training,
                                  scope='dgcnn_fc2', bn_decay=bn_decay)

    print "##########fc2", net
    # net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
    #                       scope='dp1')
    # net = tf.contrib.layers.unit_norm(net, dim=1)  # to have unit activation to last layer

    net, out_weight, out_biases = tf_util.fully_connected(net, num_point*3*4, activation_fn=None, scope='dgcnn_output')

    net_recon = tf.reshape(net, (batch_size, num_point * 4, 3))

    # 6d pose
    net_rot, _, _ = tf_util.fully_connected(end_points['embedding'], 512, bn=True, is_training=is_training,
                                        scope='dgcnn_rot_fc1', bn_decay=bn_decay)
    net_rot, _, _ = tf_util.fully_connected(net_rot, 256, bn=True, is_training=is_training,
                                            scope='dgcnn_rot_fc2', bn_decay=bn_decay)
    net_rot, _, _ = tf_util.fully_connected(net_rot, 3, activation_fn=None, scope='dgcnn_output_rot')

    net_trans, _, _ = tf_util.fully_connected(end_points['embedding'], 512, bn=True, is_training=is_training,
                                            scope='dgcnn_trans_fc1', bn_decay=bn_decay)
    net_trans, _, _ = tf_util.fully_connected(net_trans, 256, bn=True, is_training=is_training,
                                            scope='dgcnn_trans_fc2', bn_decay=bn_decay)
    net_trans, _, _ = tf_util.fully_connected(net_trans, 3, activation_fn=None, scope='dgcnn_output_trans')

    return net_recon, net_rot, net_trans, end_points


def get_model_dgcnn_mean_vae(point_cloud, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    input_image = tf.expand_dims(point_cloud, -1)

    print "###########point_cloud", point_cloud

    k = 10
    adj_matrix = tf_util.pairwise_xyz_distance(point_cloud)
    nn_idx = tf_util.knn(adj_matrix, k=k)
    edge_feature = tf_util.get_edge_feature(point_cloud, nn_idx=nn_idx, k=k)

    print "##########edge_feature", edge_feature

    net = tf_util.conv2d(edge_feature, 64, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='dgcnn1', bn_decay=bn_decay)

    print "##########dgcnn1", net

    net = tf.reduce_mean(net, axis=-2, keep_dims=True)
    net1 = net

    print "#########reduce_mean net1", net1

    adj_matrix = tf_util.pairwise_xyz_distance(net)
    nn_idx = tf_util.knn(adj_matrix, k=k)
    edge_feature = tf_util.get_edge_feature(net, nn_idx=nn_idx, k=k)

    print "###########edge feature", edge_feature

    net = tf_util.conv2d(edge_feature, 64, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='dgcnn2', bn_decay=bn_decay)

    print "###########dgcnn2", net

    net = tf.reduce_mean(net, axis=-2, keep_dims=True)
    net2 = net

    print "###########reduce_mean net2", net2

    adj_matrix = tf_util.pairwise_xyz_distance(net)
    nn_idx = tf_util.knn(adj_matrix, k=k)
    edge_feature = tf_util.get_edge_feature(net, nn_idx=nn_idx, k=k)

    print "###########edge feature", edge_feature

    net = tf_util.conv2d(edge_feature, 64, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='dgcnn3', bn_decay=bn_decay)

    print "###########dgcnn3", net

    net = tf.reduce_mean(net, axis=-2, keep_dims=True)
    net3 = net

    print "###########reduce_mean net3", net3

    adj_matrix = tf_util.pairwise_xyz_distance(net)
    nn_idx = tf_util.knn(adj_matrix, k=k)
    edge_feature = tf_util.get_edge_feature(net, nn_idx=nn_idx, k=k)

    print "###########edge feature", edge_feature

    net = tf_util.conv2d(edge_feature, 128, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='dgcnn4', bn_decay=bn_decay)

    print "###########dgcnn4", net

    net = tf.reduce_mean(net, axis=-2, keep_dims=True)
    net4 = net

    print "###########reduce_mean net4", net4

    net = tf_util.conv2d(tf.concat([net1, net2, net3, net4], axis=-1), 1024, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='dgcnn_agg', bn_decay=bn_decay)

    max_indices = tf.argmax(net, axis=1)

    print "############agg", net

    net = tf.reduce_mean(net, axis=1, keep_dims=True)

    print "###########reduce_mean", net

    # MLP on global point cloud vector
    net = tf.reshape(net, [batch_size, -1])

    z_mean, _, _ = tf_util.fully_connected(net, 1024, bn=True, is_training=is_training,
                                           scope='dgcnn_z_mean', bn_decay=bn_decay)

    z_std, _, _ = tf_util.fully_connected(net, 1024, bn=True, is_training=is_training,
                                          scope='dgcnn_z_std', bn_decay=bn_decay)

    net = z_mean + z_std * tf.random_normal(tf.shape(z_mean), 0, 1, dtype=tf.float32)

    end_points['embedding'] = net
    print "###########bottle neck", net

    net, _, _ = tf_util.fully_connected(net, 1024, bn=True, is_training=is_training,
                                  scope='dgcnn_fc1', bn_decay=bn_decay)
    print "##########fc1", net

    net, _, _ = tf_util.fully_connected(net, 1024, bn=True, is_training=is_training,
                                  scope='dgcnn_fc2', bn_decay=bn_decay)

    print "##########fc2", net
    # net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
    #                       scope='dp1')
    # net = tf.contrib.layers.unit_norm(net, dim=1)  # to have unit activation to last layer

    net, out_weight, out_biases = tf_util.fully_connected(net, num_point*3*4, activation_fn=None, scope='dgcnn_output')

    net = tf.reshape(net, (batch_size, num_point * 4, 3))

    return net, z_mean, z_std, end_points