import argparse
import math
import numpy as np
import tensorflow as tf
import importlib
import os
import sys
import open3d
import transforms3d
import matplotlib.pyplot as plt
import random
import scipy.io
import psutil

# bug: when batch size is 1, prediction is always 0??????


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'losses'))
sys.path.append(os.path.join(BASE_DIR, 'ycb_video_data'))
import trans_distance
import angular_distance_taylor
# import obj_cor_distance
import classification_loss
from datetime import datetime
import multiprocessing
sys.path.append(os.path.join(BASE_DIR, 'test_scripts'))
from coord_free_rot_decomp_tf import rotation_decomp
from hidden_point_removal import sphericalFlip_org, hidden_point_removal_org
import chamfer_loss

# from data_gen_tools import *

class_names = ["00_master_chef_can", "01_cracker_box", "02_sugar_box", "03_tomato_soup_can", "04_mustard_bottle", "05_tuna_fish_can", "06_pudding_box",
               "07_gelatin_box", "08_potted_meat_can", "09_banana", "10_pitcher_base", "11_bleach_cleanser", "12_bowl", "13_mug",
               "14_power_drill", "15_wood_block", "16_scissors", "17_large_marker", "18_large_clamp", "19_extra_large_clamp", "20_foam_brick"]
NUM_CLASS = 21

# Global settings
# cvpc4
#data_dir = '/data_c/PointNet/pointnet/ycb_video_data/tfRecords/FPS1024/'
#data_dir = '/data_c/PointNet/pointnet/ycb_video_data/tfRecords/FPS1024_only_real/'
#object_model_dir = "/data_c/YCB_Video_Dataset/tfrecords/obj_models.tfrecords"
# cvgpu1
data_dir = '/data/ge/PointNet/pointnet/ycb_video_data/tfRecords/'
# data_dir = '/data/ge/PointNet/pointnet/ycb_video_data/tfRecords/FPS1024_only_real/'
object_model_dir = "/data/ge/YCB_Video_Dataset/tfrecords/obj_models.tfrecords"
# cvpc7
#data_dir = "/data1/ge/PointNet/pointnet/ycb_video_data/tfRecords/"
#object_model_dir = "/data1/ge/PointNet/pointnet/ycb_video_data/tfRecords/obj_models.tfrecords"


# train_file_lists = []
# valid_file_lists = []

target_cls = np.arange(21)
# target_cls = [1]

b_visual = False

r_0 = np.array([1, 0, 0])
r_1 = np.array([0, 1, 0])
r_2 = np.array([0, 0, 1])
sym_axis_list = []
sym_axis_list.append(r_0)
sym_axis_list.append(r_1)
sym_axis_list.append(r_2)
sym_axis = np.asarray(sym_axis_list)


train_filenames = []
# valid_filenames = []
for cls in target_cls:
    for i in range(1):
        train_filename = data_dir + "train_files_FPS1024_" + str(cls) + "_0.tfrecords"
        train_filenames.append(train_filename)
        train_filename = data_dir + "valid_files_FPS1024_" + str(cls) + "_0.tfrecords"
        train_filenames.append(train_filename)


def decode(serialized_example, total_num_point):
    features = tf.parse_example(
        [serialized_example],
        features={
            'xyz': tf.FixedLenFeature([total_num_point, 3], tf.float32),
            'rgb': tf.FixedLenFeature([total_num_point, 3], tf.float32),
            'translation': tf.FixedLenFeature([3], tf.float32),
            'quaternion': tf.FixedLenFeature([4], tf.float32),
            'num_valid_points_in_segment': tf.FixedLenFeature([], tf.int64),
            'seq_id': tf.FixedLenFeature([], tf.int64),
            'frame_id': tf.FixedLenFeature([], tf.int64),
            'class_id': tf.FixedLenFeature([], tf.int64)
        })
    return features


def quat2axag(quat):
    w = quat[:, 0]
    x = quat[:, 1]
    y = quat[:, 2]
    z = quat[:, 3]
    len2 = x * x + y * y + z * z
    theta = 2 * tf.acos(tf.maximum(tf.minimum(w, 1), -1))
    ax = tf.stack([x, y, z], axis=1)
    ax = ax / tf.expand_dims(tf.math.sqrt(len2),1)
    ag = theta
    axag = ax * tf.expand_dims(ag, 1)
    return axag


def quat2axag_ds(x):
    x['axisangle'] = quat2axag(x['quaternion'])
    return x


def read_and_decode_obj_model(filename):
    models = []
    labels = []

    features = {'label': tf.FixedLenFeature([], tf.int64),
                'model': tf.FixedLenFeature([2048, 6], tf.float32)}

    for examples in tf.python_io.tf_record_iterator(filename):
        example = tf.parse_single_example(examples, features=features)
        models.append(example['model'])
        labels.append(example['label'])

    return models, labels


def get_object_model(x):

    obj_model, _ = read_and_decode_obj_model(object_model_dir)
    obj_model_tf = tf.convert_to_tensor(obj_model)
    x['obj_model'] = obj_model_tf

    x['obj_batch'] = tf.gather(obj_model_tf, x['class_id'], axis=0)

    return x


def get_rotation_matrix(x):
    x['axisangle'] = tf.dtypes.cast(x['axisangle'], dtype=tf.float64)
    rot_gt_mat = angular_distance_taylor.exponential_map(x['axisangle'])

    x['rot_mat'] = tf.dtypes.cast(rot_gt_mat, dtype=tf.float32)

    return x


def transform_object_model(x):

    model_xyz_rot = tf.matmul(x['obj_batch'][:, :, 0:3], tf.transpose(x['rot_mat'], perm=[0, 2, 1]))
    x['model_xyz_rot_trans'] = model_xyz_rot + tf.tile(tf.expand_dims(x['translation'], 1), [1, 2048, 1])

    return x


def get_small_data(dataset, batch_size, total_num_point):
    ncores = psutil.cpu_count()
    dataset = dataset.map(lambda x: decode(x, total_num_point))
    dataset = dataset.map(quat2axag_ds)
    dataset = dataset.map(get_object_model)
    dataset = dataset.map(get_rotation_matrix)
    dataset = dataset.map(transform_object_model)
    dataset = dataset.map(lambda x: sphericalFlip_org(x,
                                                      tf.zeros_like(x['translation']),
                                                      tf.tile(tf.constant([[0.8 * math.pi]]), [1, 1])))

    dataset = dataset.map(hidden_point_removal_org, num_parallel_calls=ncores)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(1)

    return dataset


def reshape_element(element, batch_size, total_num_point):
    element['xyz'] = tf.reshape(element['xyz'], [batch_size, total_num_point, 3])
    element['rgb'] = tf.reshape(element['rgb'], [batch_size, total_num_point, 3])
    element['translation'] = tf.reshape(element['translation'], [batch_size, 3])
    element['quaternion'] = tf.reshape(element['quaternion'], [batch_size, 4])
    element['class_id'] = tf.reshape(element['class_id'], [batch_size])
    element['num_valid_points_in_segment'] = tf.reshape(element['num_valid_points_in_segment'], [batch_size])

    return element


# ==============================================================================


def log_string(out_str, dir):
    dir.write(out_str + '\n')
    dir.flush()
    print(out_str)


# define the graph
def setup_graph(general_opts, train_opts, hyperparameters):
    tf.reset_default_graph()
    now = datetime.now()

    BATCH_SIZE = hyperparameters['batch_size']
    NUM_POINT = general_opts['num_point']
    TOTAL_NUM_POINT = general_opts['total_num_point']
    MAX_EPOCH = train_opts['max_epoch']
    BASE_LEARNING_RATE = hyperparameters['learning_rate']
    GPU_INDEX = general_opts['gpu']
    OPTIMIZER = train_opts['optimizer']
    MODEL = importlib.import_module(general_opts['model'])  # import network module
    MODEL_FILE = os.path.join(BASE_DIR, 'models', general_opts['model'] + '.py')
    CURRENT_FILE = os.path.realpath(__file__)
    minimum_points_in_segment = NUM_POINT
    K_NEIGHBOR = hyperparameters['k_neighbor']

    LOG_DIR = general_opts['log_dir'] + "/" + str(NUM_CLASS) + "/" + "6d/" + now.strftime("%Y%m%d-%H%M%S") + "/"
    if not os.path.exists(LOG_DIR):
        os.mkdir(LOG_DIR)
    LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
    LOG_FOUT.write(str(general_opts) + '\n')
    LOG_FOUT.write(str(train_opts) + '\n')
    LOG_FOUT.write(str(hyperparameters) + '\n')

    tf.set_random_seed(123456789)

    os.system('cp %s %s' % (MODEL_FILE, LOG_DIR))  # bkp of model def
    os.system('cp %s %s' % (CURRENT_FILE, LOG_DIR))  # bkp of train procedure

    # double check
    BN_INIT_DECAY = 0.5
    BN_DECAY_DECAY_RATE = 0.5
    BN_DECAY_DECAY_STEP = float(40)
    BN_DECAY_CLIP = 0.99

    with tf.Graph().as_default():

        with tf.device('/cpu:0'):

            with tf.name_scope('prepare_data'):
                tr_dataset = tf.data.TFRecordDataset(train_filenames).shuffle(1000000)
                tr_dataset = get_small_data(tr_dataset, batch_size=BATCH_SIZE, total_num_point=TOTAL_NUM_POINT)
                tr_iterator = tr_dataset.make_initializable_iterator()

                iter_handle = tf.placeholder(tf.string, shape=[], name='iterator_handle')
                iterator = tf.data.Iterator.from_string_handle(iter_handle, tr_dataset.output_types,
                                                               tr_dataset.output_shapes)
                next_element = iterator.get_next()
                next_element = reshape_element(next_element, batch_size=BATCH_SIZE, total_num_point=TOTAL_NUM_POINT)

                obj_model, _ = read_and_decode_obj_model(object_model_dir)
                obj_model_tf = tf.convert_to_tensor(obj_model)

        with tf.device('/gpu:' + str(GPU_INDEX)):

            is_training_pl_encoder = tf.placeholder(tf.bool, shape=())
            is_training_pl = tf.placeholder(tf.bool, shape=())

            batch = tf.Variable(0.)

            bn_momentum = tf.train.exponential_decay(
                BN_INIT_DECAY,
                batch * BATCH_SIZE,
                BN_DECAY_DECAY_STEP,
                BN_DECAY_DECAY_RATE,
                staircase=True)
            bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)

            tf.summary.scalar('bn_decay', bn_decay)

            obj_batch = tf.gather(obj_model_tf, next_element['class_id'], axis=0)
            obj_batch = obj_batch[:, 0:2048, :]

            next_element_xyz = next_element['xyz'][:, 0:NUM_POINT, :]
            next_element_rgb = next_element['rgb'][:, 0:NUM_POINT, :]
            # next_element_yiq = tf.image.rgb_to_yiq(next_element_rgb)

            cls_gt_onehot = tf.one_hot(indices=next_element['class_id'], depth=len(target_cls))
            cls_gt_onehot_expand = tf.expand_dims(cls_gt_onehot, axis=1)
            cls_gt_onehot_tile = tf.tile(cls_gt_onehot_expand, [1, NUM_POINT, 1])

            current_batch_axag = tf.squeeze(next_element['axisangle'], 1)
            current_batch_axag = tf.dtypes.cast(current_batch_axag, dtype=tf.float64)

            visiblePoints_org = tf.reshape(next_element['visiblePoints_org'], [BATCH_SIZE, 2048 + 1, 3])  # no occluder
            visiblePoints_org_final = visiblePoints_org[:, 0:NUM_POINT * 4, :]

            # ignore data augmentation
            xyz_graph_input = next_element_xyz
            trans_gt_graph_input = next_element['translation']
            rot_gt_graph_input = tf.cast(current_batch_axag, tf.float64)

            with tf.name_scope('6d_pose'):
                element_mean = tf.reduce_mean(xyz_graph_input, axis=1)

                xyz_normalized = xyz_graph_input - tf.expand_dims(element_mean, 1)

                xyz_recon_res, rot_pred, trans_pred_res, endpoint = MODEL.get_model_dgcnn_mean_6d(
                    tf.concat([xyz_normalized, cls_gt_onehot_tile], axis=2),
                    is_training_pl_encoder, is_training_pl, K_NEIGHBOR, bn_decay=bn_decay)

                xyz_recon = xyz_recon_res + tf.tile(tf.expand_dims(element_mean, 1), [1, xyz_recon_res.shape[1], 1])
                trans_pred = trans_pred_res + element_mean

                xyz_loss, xyz_loss_per_sample = chamfer_loss.get_loss(xyz_recon, visiblePoints_org_final)
                tf.summary.scalar('chamfer_loss', xyz_loss)

            with tf.name_scope('translation'):

                trans_loss, trans_loss_perSample = trans_distance.get_translation_error(trans_pred,
                                                                                        trans_gt_graph_input)
                tf.summary.scalar('trans_loss', trans_loss)
                tf.summary.scalar('trans_loss_min', tf.reduce_min(trans_loss_perSample))
                tf.summary.scalar('trans_loss_max', tf.reduce_max(trans_loss_perSample))

            with tf.name_scope('rotation'):

                rot_pred = tf.cast(rot_pred, tf.float64)

                axag_loss, axag_loss_perSample = angular_distance_taylor.get_rotation_error(rot_pred,
                                                                                            rot_gt_graph_input)
                axag_loss = tf.cast(axag_loss, tf.float32)

                tf.summary.scalar('axag_loss', axag_loss)
                tf.summary.scalar('axag_loss_min', tf.reduce_min(axag_loss_perSample))
                tf.summary.scalar('axag_loss_max', tf.reduce_max(axag_loss_perSample))

            with tf.name_scope('rotation_decomp'):
                # gt quaternion to mat
                current_gt_mat = angular_distance_taylor.exponential_map(current_batch_axag)
                # pred axag to mat
                current_pred_mat = angular_distance_taylor.exponential_map(rot_pred)
                # rotation decompose
                sym_axis_tf = tf.convert_to_tensor(sym_axis, dtype=tf.float64)
                theta_gt = rotation_decomp(current_gt_mat, sym_axis_tf, BATCH_SIZE)
                theta_pred = rotation_decomp(current_pred_mat, sym_axis_tf, BATCH_SIZE)
                decomp_3_loss_perSample = tf.math.abs(theta_gt - theta_pred) # abs or not??????

                decomp_3_loss = tf.reduce_mean(decomp_3_loss_perSample, axis=0)

                decomp_loss_perSample = tf.reduce_sum(tf.math.abs(theta_gt - theta_pred), axis=1)
                decomp_loss = tf.reduce_mean(decomp_loss_perSample)

                tf.summary.scalar('decomp_loss', decomp_loss)
                tf.summary.scalar('decomp_loss_x', decomp_3_loss[0])
                tf.summary.scalar('decomp_loss_y', decomp_3_loss[1])
                tf.summary.scalar('decomp_loss_z', decomp_3_loss[2])
                tf.summary.scalar('decomp_loss_min', tf.reduce_min(decomp_loss_perSample))
                tf.summary.scalar('decomp_loss_max', tf.reduce_max(decomp_loss_perSample))

            learning_rate = BASE_LEARNING_RATE

            tf.summary.scalar('learning_rate', learning_rate)

            if OPTIMIZER == 'gd':
                optimizer = tf.train.GradientDescentOptimizer(learning_rate*10)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)

            total_loss = 1000 * xyz_loss + 10 * trans_loss + axag_loss

            tf.summary.scalar('total_loss', total_loss)

            # reference: http://matpalm.com/blog/viz_gradient_norms/
            gradients = optimizer.compute_gradients(loss=total_loss)
            train_op = optimizer.apply_gradients(gradients, global_step=batch)

            # Add ops to save and restore all the variables.
            # saver = tf.train.Saver(max_to_keep=None)

            variables_to_restore = [v for v in tf.global_variables() if v.name.split('/')[0] in ['dgcnn1',
                                                                                                 'dgcnn2',
                                                                                                 'dgcnn3',
                                                                                                 'dgcnn4',
                                                                                                 'dgcnn_agg']]
            variables_to_initialize = [v for v in tf.global_variables() if v.name.split('/')[0] not in ['dgcnn1',
                                                                                                 'dgcnn2',
                                                                                                 'dgcnn3',
                                                                                                 'dgcnn4',
                                                                                                 'dgcnn_agg']]
            saver = tf.train.Saver(variables_to_restore) # to load partial weights
            saver_all = tf.train.Saver()  # to save all weights

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        # config.gpu_options.per_process_gpu_memory_fraction = 0.8
        sess = tf.Session(config=config)
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        # sess = tf_debug.TensorBoardDebugWrapperSession(sess, 'localhost:6064')

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                                             sess.graph)

        # Init variables
        # init = tf.global_variables_initializer()
        init = tf.initializers.variables(var_list=variables_to_initialize)

        sess.run(init, {is_training_pl_encoder: False, is_training_pl: True})

        # Restore variables from disk.
        saver.restore(sess, '/data/ge/PointNet/pointnet/log/21/6d/20200903-203813/model.ckpt')
        print "Model restored."
        saver = saver_all

        ops = {'is_training_pl_encoder': is_training_pl_encoder,
               'is_training_pl': is_training_pl,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'trans_loss': trans_loss,
               'trans_loss_perSample': trans_loss_perSample,
               'axag_loss': axag_loss,
               'axag_loss_perSample': axag_loss_perSample,
               'translation': next_element['translation'],
               'quaternion': next_element['quaternion'],
               'trans_gt_graph_input': trans_gt_graph_input,
               'rot_gt_graph_input': rot_gt_graph_input,
               'current_batch_axag': current_batch_axag,
               'xyz': next_element_xyz,
               'rgb': next_element_rgb,
               'class_id': next_element['class_id'],
               'seq_id': next_element['seq_id'],
               'frame_id': next_element['frame_id'],
               'obj_batch': obj_batch,
               'num_valid_points_in_segment': next_element['num_valid_points_in_segment'],
               'rot_pred': rot_pred,
               'trans_pred': trans_pred,
               'current_gt_mat': current_gt_mat,
               'decomp_3_loss_perSample': decomp_3_loss_perSample,
               'decomp_3_loss': decomp_3_loss,
               'xyz_loss': xyz_loss,
               'xyz_loss_per_sample': xyz_loss_per_sample,
               'visiblePoints_org_final': visiblePoints_org_final,
               'handle': iter_handle}

        count_perClass = np.zeros([NUM_CLASS], dtype=np.int32)
        xyz_loss_perClass = [[] for _ in range(NUM_CLASS)]
        axag_loss_perClass = [[] for _ in range(NUM_CLASS)]
        trans_loss_perClass = [[] for _ in range(NUM_CLASS)]
        decomp_loss_x_perClass = [[] for _ in range(NUM_CLASS)]
        decomp_loss_y_perClass = [[] for _ in range(NUM_CLASS)]
        decomp_loss_z_perClass = [[] for _ in range(NUM_CLASS)]

        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch), dir=LOG_FOUT)
            sys.stdout.flush()

            training_handle = sess.run(tr_iterator.string_handle())
            # validation_handle = sess.run(val_iterator.string_handle())

            sess.run(tr_iterator.initializer)
            # print(tf.trainable_variables())

            train_graph(sess, ops, train_writer, train_writer, training_handle, epoch,
                        count_perClass, xyz_loss_perClass, axag_loss_perClass, trans_loss_perClass,
                        decomp_loss_x_perClass, decomp_loss_y_perClass, decomp_loss_z_perClass,
                        logFOut=LOG_FOUT, num_point=NUM_POINT, batch_size=BATCH_SIZE, logdir=LOG_DIR, saver=saver)


def train_graph(sess, ops, train_writer, val_writer, training_handle, epoch, count_perClass, xyz_loss_perClass,
                axag_loss_perClass, trans_loss_perClass,
                decomp_loss_x_perClass, decomp_loss_y_perClass, decomp_loss_z_perClass,
                logFOut, num_point, batch_size, logdir, saver):
    """ ops: dict mapping from string to tf ops """
    print "==================train======================"
    is_training_encoder = False
    is_training = True

    # measure duration of each subprocess
    start_time = datetime.now()
    summary1 = tf.Summary()

    batch_idx = 0

    axag_loss_perClass_val = [[] for _ in range(NUM_CLASS)]
    trans_loss_perClass_val = [[] for _ in range(NUM_CLASS)]
    mean_dist_loss_perClass_val = [[] for _ in range(NUM_CLASS)]

    while True:
        try:
            total_seen = 0

            # evaluate ever 100 batch during training
            if batch_idx != 0 and batch_idx % 2000 == 0:
                # sess.run(val_iterator.initializer)
                # trans_loss_valid, axag_loss_valid = eval_graph(sess, ops, val_writer, validation_handle, batch_idx, epoch,
                #                                      logFOut, num_point, batch_size, axag_loss_perClass_val, trans_loss_perClass_val,
                #                                                mean_dist_loss_perClass_val)
                # model_dir = "model_" + str(epoch) + "_" + str(batch_idx) + "_" + str(trans_loss_valid) + "_" + str(axag_loss_valid) + ".ckpt"
                if epoch % 20 == 0:
                    model_dir = "model_" + str(epoch) + ".ckpt"
                else:
                    model_dir = "model.ckpt"
                save_path = saver.save(sess, os.path.join(logdir, model_dir))
                print "Model saved in file: %s" % save_path

            feed_dict = {ops['is_training_pl_encoder']: is_training_encoder, ops['is_training_pl']: is_training, ops['handle']: training_handle}

            summary, step, _, xyz, rgb, class_id, seq_id, frame_id, quaternion, translation, \
            trans_gt_graph_input, rot_gt_graph_input, obj_batch, \
            trans_loss, trans_loss_perSample, axag_loss, axag_loss_perSample,\
            current_batch_axag, num_valid_points_in_segment, rot_pred, \
            current_gt_mat, decomp_3_loss_perSample, decomp_3_loss, xyz_loss, xyz_loss_per_sample, \
            visiblePoints_org_final = sess.run(
                [ops['merged'],
                 ops['step'],
                 ops['train_op'],
                 ops['xyz'],
                 ops['rgb'],
                 # ops['yiq'],
                 ops['class_id'],
                 ops['seq_id'],
                 ops['frame_id'],
                 ops['quaternion'],
                 ops['translation'],
                 ops['trans_gt_graph_input'],
                 ops['rot_gt_graph_input'],
                 # ops['trans_aug'],
                 # ops['rot_aug_mat'],
                 ops['obj_batch'],
                 ops['trans_loss'],
                 ops['trans_loss_perSample'],
                 ops['axag_loss'],
                 ops['axag_loss_perSample'],
                 ops['current_batch_axag'],
                 ops['num_valid_points_in_segment'],
                 ops['rot_pred'],
                 ops['current_gt_mat'],
                 ops['decomp_3_loss_perSample'],
                 ops['decomp_3_loss'],
                 ops['xyz_loss'],
                 ops['xyz_loss_per_sample'],
                 ops['visiblePoints_org_final']
                 ],
                feed_dict=feed_dict)

            # print xyz.shape
            # sys.exit()
            # with open('/data_c/6d_demo/banana_data_train.txt', 'ab') as f:
            #     np.savetxt(f, np.mean(np.squeeze(xyz), axis=0), fmt='%.4f', delimiter=',', newline=' ')
            #     f.write(b'\n')

            # print step
            # # sanity check
            if b_visual:
                print class_id
                batch_sample_idx = 6
                current_quat = quaternion[batch_sample_idx]
                current_ax, current_ag = transforms3d.quaternions.quat2axangle(current_quat)

                # sys.exit()
                #
                # points = [[0,0,0], current_ax]
                # line = [[0,1]]
                # line_set = open3d.LineSet()
                # line_set.points = open3d.Vector3dVector(points)
                # line_set.lines = open3d.Vector2iVector(line)
                # line_set.colors = open3d.Vector3dVector([[1, 0, 0]])
                #
                # From camera to object
                rotmat = transforms3d.axangles.axangle2mat(current_ax, current_ag)
                sym_ax = np.array([0., 0.,1.])
                sym_ag = 0
                sym_mat = transforms3d.axangles.axangle2mat(sym_ax, sym_ag)
                rotmat_sym = np.matmul(rotmat, sym_mat)
                xyz_remove_rot = np.dot(xyz[batch_sample_idx,:,:], rotmat_sym)
                xyz_remove_trans = xyz_remove_rot - np.dot(rotmat_sym.T, translation[batch_sample_idx,:])

                # segment_ptCloud_org = open3d.PointCloud()
                # segment_ptCloud_org.points = open3d.Vector3dVector(xyz[batch_sample_idx,...])
                # segment_ptCloud_org.paint_uniform_color([0.1, 0.9, 0.1])
                #
                # xyz_aug_remove_rot = np.dot(xyz_w_pose_aug[batch_sample_idx, :, :], rot_aug_mat[batch_sample_idx, ...])
                # xyz_aug_remove_trans = xyz_aug_remove_rot - np.dot(rot_aug_mat[batch_sample_idx, ...].T, trans_aug[batch_sample_idx,...])

                segment_ptCloud = open3d.geometry.PointCloud()
                segment_ptCloud.points = open3d.utility.Vector3dVector(xyz[batch_sample_idx,:,:])
                segment_ptCloud.colors = open3d.utility.Vector3dVector(rgb[batch_sample_idx, :, :])

                model_pCloud = open3d.geometry.PointCloud()
                model_pCloud.points = open3d.utility.Vector3dVector(obj_batch[batch_sample_idx, :, 0:3])
                model_pCloud.colors = open3d.utility.Vector3dVector(obj_batch[batch_sample_idx, :, 3:6])
                model_pCloud.paint_uniform_color([0.1, 0.9, 0.1])

                visiblefull_ptCloud = open3d.geometry.PointCloud()
                visiblefull_ptCloud.points = open3d.utility.Vector3dVector(visiblePoints_org_final[batch_sample_idx, :, :])
                visiblefull_ptCloud.paint_uniform_color([0.9, 0.1, 0.1])

                mesh_frame = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15, origin=[0, 0, 0])
                open3d.visualization.draw_geometries([mesh_frame, segment_ptCloud, model_pCloud, visiblefull_ptCloud])

            # for each sample in current batch
            for x, y, z, rx, ry, rz, c in zip(xyz_loss_per_sample, axag_loss_perSample, trans_loss_perSample,
                                              decomp_3_loss_perSample[:, 0], decomp_3_loss_perSample[:, 1], decomp_3_loss_perSample[:, 2],
                                              class_id):
                xyz_loss_perClass[c].append(x)
                axag_loss_perClass[c].append(y)
                trans_loss_perClass[c].append(z)
                decomp_loss_x_perClass[c].append(rx)
                decomp_loss_y_perClass[c].append(ry)
                decomp_loss_z_perClass[c].append(rz)


            print "epoch %d batch %d xyz_loss %f trans_loss %f axag_loss %f euler x loss %f euler y loss %f euler z loss %f" \
                  % (epoch, batch_idx, xyz_loss, trans_loss, axag_loss, decomp_3_loss[0], decomp_3_loss[1], decomp_3_loss[2])

            # write to tensorboard
            if batch_idx != 0 and batch_idx % 100 == 0:
                for i in target_cls:
                    count_perClass[i] = count_perClass[i] + len(axag_loss_perClass[i])
                    avg_xyz_loss = np.average(xyz_loss_perClass[i])
                    avg_axag_loss = np.average(axag_loss_perClass[i])
                    avg_trans_loss = np.average(trans_loss_perClass[i])
                    avg_decomp_loss_x = np.average(decomp_loss_x_perClass[i])
                    avg_decomp_loss_y = np.average(decomp_loss_y_perClass[i])
                    avg_decomp_loss_z = np.average(decomp_loss_z_perClass[i])
                    summary1.value.add(tag="xyz_loss_per_class_train/" + class_names[i], simple_value=avg_xyz_loss)
                    summary1.value.add(tag="axag_loss_per_class_train/"+class_names[i], simple_value=avg_axag_loss)
                    summary1.value.add(tag="trans_loss_per_class_train/"+class_names[i], simple_value=avg_trans_loss)
                    summary1.value.add(tag="sample_count_per_class_train/"+class_names[i], simple_value=count_perClass[i])
                    summary1.value.add(tag="decomp_loss_x_per_class_train/" + class_names[i], simple_value=avg_decomp_loss_x)
                    summary1.value.add(tag="decomp_loss_y_per_class_train/" + class_names[i], simple_value=avg_decomp_loss_y)
                    summary1.value.add(tag="decomp_loss_z_per_class_train/" + class_names[i], simple_value=avg_decomp_loss_z)

                    xyz_loss_perClass[i] = []
                    axag_loss_perClass[i] = []
                    trans_loss_perClass[i] = []
                    decomp_loss_x_perClass[i] = []
                    decomp_loss_y_perClass[i] = []
                    decomp_loss_z_perClass[i] = []

                train_writer.add_summary(summary, step)
                train_writer.add_summary(summary1, step)

            total_seen += batch_size
            batch_idx = batch_idx + 1

        except tf.errors.OutOfRangeError:
            print('End of data!')
            model_dir = "model.ckpt"
            save_path = saver.save(sess, os.path.join(logdir, model_dir))
            print "Model saved in file: %s" % save_path
            break

    time_elapsed = datetime.now() - start_time
    out_str = 'Current epoch Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed)
    logFOut.write(out_str + '\n')
    logFOut.flush()
    print(out_str)


def get_training_argparser():
    parser = argparse.ArgumentParser()

    general = parser.add_argument_group('general')
    general.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
    general.add_argument('--model', default='pointnet_ycb_23_decoder_4',
                         help='Model name: pointnet_cls or pointnet_cls_basic or pointnet_cls_rot_basic [default: pointnet_cls]')
    general.add_argument('--log_dir', default='log', help='Log dir [default: log]')
    general.add_argument('--num_point', type=int, default=256, help='Point Number [256/512/1024] [default: 256]')
    general.add_argument('--total_num_point', type=int, default=1024, help='Dataset Point Number [256/512/1024] [default: 1024]')

    train_opts = parser.add_argument_group('training_options')
    train_opts.add_argument('--max_epoch', type=int, default=90, help='Epoch to run [default: 100]')
    train_opts.add_argument('--optimizer', default='adam', help='adam or gd [default: adam]')

    hyperparameters = parser.add_argument_group('hyperparameters')
    hyperparameters.add_argument('--batch_size', type=int, default=128,
                                 help='Batch Size during training [default: 128]')
    hyperparameters.add_argument('--learning_rate', type=float, default=0.0008,
                                 help='Initial learning rate [default: 0.0008]')
    hyperparameters.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
    hyperparameters.add_argument('--decay_step', type=int, default=30000,
                                 help='Decay step for lr decay [default: 30000]')
    hyperparameters.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
    hyperparameters.add_argument('--trans_tol', type=float, default=0.1,
                                 help='Translation error tolerance [default:0.05]')
    hyperparameters.add_argument('--k_neighbor', type=int, default=10,
                                 help='# of neighbors for dgcnn [default:10]')

    return parser


def parse_arg_groups(parser):
    args = parser.parse_args()
    arg_groups = {}
    for group in parser._action_groups:
        arg_groups[group.title] = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
    return arg_groups


if __name__ == "__main__":
    parser = get_training_argparser()
    arg_groups = parse_arg_groups(parser)
    general_opts, train_opts, hyperparameters = arg_groups['general'], arg_groups['training_options'], arg_groups[
        'hyperparameters']
    setup_graph(general_opts=general_opts,
                train_opts=train_opts,
                hyperparameters=hyperparameters)
    # LOG_FOUT.close()
