import argparse
import math
import numpy as np
import tensorflow as tf
import importlib
import os
import sys
import open3d
import psutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'losses'))
sys.path.append(os.path.join(BASE_DIR, 'ycb_video_data'))
sys.path.append(os.path.join(BASE_DIR, 'tf_ops/sampling'))
import trans_distance
import angular_distance_taylor
from datetime import datetime
from hidden_point_removal import sphericalFlip, hidden_point_removal, sphericalFlip_org, hidden_point_removal_org
from generate_occluder import get_random_object_occluder, get_random_spherical_occluder
import chamfer_loss


class_names = ["00_master_chef_can", "01_cracker_box", "02_sugar_box", "03_tomato_soup_can", "04_mustard_bottle", "05_tuna_fish_can", "06_pudding_box",
               "07_gelatin_box", "08_potted_meat_can", "09_banana", "10_pitcher_base", "11_bleach_cleanser", "12_bowl", "13_mug",
               "14_power_drill", "15_wood_block", "16_scissors", "17_large_marker", "18_large_clamp", "19_extra_large_clamp", "20_foam_brick"]
NUM_CLASS = 21

object_model_dir = "object_model_tfrecord/obj_models.tfrecords"
target_cls = np.arange(NUM_CLASS)

b_visual = False

train_filenames = []
for cls in target_cls:
    train_filename = "ycb_video_data_tfRecords/train_syn/" + str(cls) + "_syn.tfrecords"
    train_filenames.append(train_filename)


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


def decode(serialized_example):
    features = tf.parse_example(
        [serialized_example],
        features={
            'translation': tf.FixedLenFeature([3], tf.float32),
            'axisangle': tf.FixedLenFeature([3], tf.float32),
            'class_id': tf.FixedLenFeature([], tf.int64),
        })
    return features


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


def get_small_data(dataset, batch_size):
    ncores = psutil.cpu_count()
    dataset = dataset.map(decode)
    dataset = dataset.map(get_object_model)
    dataset = dataset.map(get_rotation_matrix)
    dataset = dataset.map(transform_object_model)
    dataset = dataset.map(lambda x: get_random_spherical_occluder(x, 'ycbv'))
    dataset = dataset.map(lambda x: sphericalFlip(x,
                                                  tf.zeros_like(x['translation']),
                                                  tf.tile(tf.constant([[0.8 * math.pi]]), [1, 1])))

    dataset = dataset.map(hidden_point_removal, num_parallel_calls=ncores)
    dataset = dataset.map(lambda x: sphericalFlip_org(x,
                                                  tf.zeros_like(x['translation']),
                                                  tf.tile(tf.constant([[0.8 * math.pi]]), [1, 1])))

    dataset = dataset.map(hidden_point_removal_org, num_parallel_calls=ncores)

    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(1)

    return dataset


def reshape_element(element, batch_size):
    element['translation'] = tf.reshape(element['translation'], [batch_size, 3])
    element['axisangle'] = tf.reshape(element['axisangle'], [batch_size, 3])
    element['class_id'] = tf.reshape(element['class_id'], [batch_size])

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
    MAX_EPOCH = train_opts['max_epoch']
    BASE_LEARNING_RATE = hyperparameters['learning_rate']
    GPU_INDEX = general_opts['gpu']
    OPTIMIZER = train_opts['optimizer']
    MODEL = importlib.import_module(general_opts['model'])  # import network module
    MODEL_FILE = os.path.join(BASE_DIR, 'models', general_opts['model'] + '.py')
    CURRENT_FILE = os.path.realpath(__file__)

    LOG_DIR = general_opts['log_dir'] + "/" + str(NUM_CLASS) + "/" + "6d/" + now.strftime("%Y%m%d-%H%M%S") + "/"
    if not os.path.exists(LOG_DIR):
        os.mkdir(LOG_DIR)
        os.mkdir(os.path.join(LOG_DIR, "recon_cloud"))
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

                full_dataset = tf.data.TFRecordDataset(train_filenames).shuffle(3042462 * 13)

                tr_dataset = full_dataset
                tr_dataset = get_small_data(tr_dataset, batch_size=BATCH_SIZE)
                tr_iterator = tr_dataset.make_initializable_iterator()

                iter_handle = tf.placeholder(tf.string, shape=[], name='iterator_handle')
                iterator = tf.data.Iterator.from_string_handle(iter_handle, tr_dataset.output_types,
                                                               tr_dataset.output_shapes)
                next_element = iterator.get_next()
                next_element = reshape_element(next_element, batch_size=BATCH_SIZE)

        with tf.device('/gpu:' + str(GPU_INDEX)):

            is_training_pl = tf.placeholder(tf.bool, shape=())
            print(is_training_pl)

            batch = tf.Variable(0.)

            bn_momentum = tf.train.exponential_decay(
                BN_INIT_DECAY,
                batch * BATCH_SIZE,
                BN_DECAY_DECAY_STEP,
                BN_DECAY_DECAY_RATE,
                staircase=True)
            bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)

            tf.summary.scalar('bn_decay', bn_decay)

            cls_gt_onehot = tf.one_hot(indices=next_element['class_id'], depth=len(target_cls))
            cls_gt_onehot_expand = tf.expand_dims(cls_gt_onehot, axis=1)
            cls_gt_onehot_tile = tf.tile(cls_gt_onehot_expand, [1, NUM_POINT, 1])

            visiblePoints = tf.reshape(next_element['visiblePoints'], [BATCH_SIZE, 2048+1+400+512-512, 3])
            visiblePoints_org = tf.reshape(next_element['visiblePoints_org'], [BATCH_SIZE, 2048+1, 3]) # no occluder

            visiblePoints_final = visiblePoints[:, 0:NUM_POINT, :]
            visiblePoints_org_final = visiblePoints_org[:, 0:NUM_POINT*4, :]

            # add point wise random noise to visiblePoint
            visiblePoints_final = visiblePoints_final + tf.random.normal(visiblePoints_final.shape, mean=0.0, stddev=0.004/3.) # 4mm
            # ref: Kinect v2 for Mobile Robot Navigation:Evaluation and Modeling, icra 2015

            model_xyz = tf.squeeze(next_element['model_xyz_rot_trans'])
            model_xyz = model_xyz[:, 0:NUM_POINT, :]

            with tf.name_scope('decoder'):
                element_mean = tf.reduce_mean(visiblePoints_final, axis=1)

                visiblePoints_final_normalized = visiblePoints_final - tf.expand_dims(element_mean, 1)

                xyz_recon_res, rot_pred, trans_pred_res, endpoint = MODEL.get_model_dgcnn_mean_6d(tf.concat([visiblePoints_final_normalized,
                                                                 cls_gt_onehot_tile], axis=2),
                                                      is_training_pl,is_training_pl,10, bn_decay=bn_decay)

                xyz_recon = xyz_recon_res + tf.tile(tf.expand_dims(element_mean, 1), [1, xyz_recon_res.shape[1], 1])
                trans_pred = trans_pred_res + element_mean

                # reconstruct complete single viewed segment in that pose
                xyz_loss, xyz_loss_per_sample = chamfer_loss.get_loss(xyz_recon, visiblePoints_org_final)
                tf.summary.scalar('chamfer_loss', xyz_loss)

            with tf.name_scope('translation'):

                trans_loss, trans_loss_perSample = trans_distance.get_translation_error(trans_pred,
                                                                                        next_element['translation'])
                tf.summary.scalar('trans_loss', trans_loss)
                tf.summary.scalar('trans_loss_min', tf.reduce_min(trans_loss_perSample))
                tf.summary.scalar('trans_loss_max', tf.reduce_max(trans_loss_perSample))

            with tf.name_scope('rotation'):

                rot_pred = tf.cast(rot_pred, tf.float64)

                axag_loss, axag_loss_perSample = angular_distance_taylor.get_rotation_error(rot_pred,
                                                                                            next_element['axisangle'])
                axag_loss = tf.cast(axag_loss, tf.float32)

                tf.summary.scalar('axag_loss', axag_loss)
                tf.summary.scalar('axag_loss_min', tf.reduce_min(axag_loss_perSample))
                tf.summary.scalar('axag_loss_max', tf.reduce_max(axag_loss_perSample))

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
            saver = tf.train.Saver(max_to_keep=None)

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                                             sess.graph)

        # Init variables
        init = tf.global_variables_initializer()

        sess.run(init, {is_training_pl: True})

        ops = {'is_training_pl': is_training_pl,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'visiblePoints_final': visiblePoints_final,
               'visiblePoints_org_final': visiblePoints_org_final,
               'class_id': next_element['class_id'],
               'xyz_loss': xyz_loss,
               'xyz_recon': xyz_recon,
               'xyz_loss_per_sample': xyz_loss_per_sample,
               'trans_loss': trans_loss,
               'trans_loss_perSample': trans_loss_perSample,
               'axag_loss': axag_loss,
               'axag_loss_perSample': axag_loss_perSample,
               'rot_pred': rot_pred,
               'trans_pred': trans_pred,
               'occluder': tf.squeeze(next_element['occluder'], 1),
               'handle': iter_handle}

        count_perClass = np.zeros([NUM_CLASS], dtype=np.int32)
        xyz_loss_perClass = [[] for _ in range(NUM_CLASS)]
        axag_loss_perClass = [[] for _ in range(NUM_CLASS)]
        trans_loss_perClass = [[] for _ in range(NUM_CLASS)]

        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch), dir=LOG_FOUT)
            sys.stdout.flush()

            training_handle = sess.run(tr_iterator.string_handle())

            sess.run(tr_iterator.initializer)

            train_graph(sess, ops, train_writer, train_writer, training_handle, epoch,
                        count_perClass, xyz_loss_perClass, axag_loss_perClass, trans_loss_perClass,
                        logFOut=LOG_FOUT, num_point=NUM_POINT, batch_size=BATCH_SIZE, logdir=LOG_DIR, saver=saver)


def train_graph(sess, ops, train_writer, val_writer, training_handle, epoch, count_perClass,
                xyz_loss_perClass, axag_loss_perClass, trans_loss_perClass, logFOut, num_point, batch_size, logdir, saver):
    """ ops: dict mapping from string to tf ops """
    print "==================train======================"
    is_training = True

    # measure duration of each subprocess
    start_time = datetime.now()
    summary1 = tf.Summary()

    batch_idx = 0

    while True:
        try:
            total_seen = 0

            feed_dict = {ops['is_training_pl']: is_training, ops['handle']: training_handle}

            summary, step, _, class_id, visiblePoints_final, visiblePoints_org_final \
            , xyz_loss, xyz_recon, xyz_loss_per_sample, \
            trans_loss, trans_loss_perSample, axag_loss, axag_loss_perSample, occluder = sess.run(
                [ops['merged'],
                 ops['step'],
                 ops['train_op'],
                 ops['class_id'],
                 ops['visiblePoints_final'],
                 ops['visiblePoints_org_final'],
                 ops['xyz_loss'],
                 ops['xyz_recon'],
                 ops['xyz_loss_per_sample'],
                 ops['trans_loss'],
                 ops['trans_loss_perSample'],
                 ops['axag_loss'],
                 ops['axag_loss_perSample'],
                 ops['occluder']
                 ],
                feed_dict=feed_dict)

            if b_visual:
                batch_sample_idx = 4

                segment_ptCloud_org = open3d.geometry.PointCloud()
                segment_ptCloud_org.points = open3d.utility.Vector3dVector(visiblePoints_org_final[batch_sample_idx,:,:])
                segment_ptCloud_org.paint_uniform_color([0.1, 0.9, 0.1])

                segment_ptCloud_recon = open3d.geometry.PointCloud()
                segment_ptCloud_recon.points = open3d.utility.Vector3dVector(xyz_recon[batch_sample_idx, :, :])
                segment_ptCloud_recon.paint_uniform_color([0.1, 0.1, 0.9])

                graph_input_ptCloud = open3d.geometry.PointCloud()
                graph_input_ptCloud.points = open3d.utility.Vector3dVector(visiblePoints_final[batch_sample_idx, :, :])
                graph_input_ptCloud.paint_uniform_color([0.9, 0.1, 0.1])

                occluder_ptCloud = open3d.geometry.PointCloud()
                occluder_ptCloud.points = open3d.utility.Vector3dVector(occluder[batch_sample_idx, :, :])
                occluder_ptCloud.paint_uniform_color([0.1, 0.1, 0.9])

                mesh_frame = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

                open3d.visualization.draw_geometries([graph_input_ptCloud, occluder_ptCloud, mesh_frame])

            # for each sample in current batch
            for x, y, z, c in zip(xyz_loss_per_sample, axag_loss_perSample, trans_loss_perSample, class_id):
                xyz_loss_perClass[c].append(x)
                axag_loss_perClass[c].append(y)
                trans_loss_perClass[c].append(z)

            print "epoch %d batch %d xyz_loss %f trans_loss %f axag_loss %f" \
                  % (epoch, batch_idx, xyz_loss, trans_loss, axag_loss)

            # write to tensorboard
            if batch_idx != 0 and batch_idx % 1000 == 0:
                for i in target_cls:
                    count_perClass[i] = count_perClass[i] + len(xyz_loss_perClass[i])
                    avg_xyz_loss = np.average(xyz_loss_perClass[i])
                    avg_axag_loss = np.average(axag_loss_perClass[i])
                    avg_trans_loss = np.average(trans_loss_perClass[i])
                    summary1.value.add(tag="xyz_loss_per_class_train/"+class_names[i], simple_value=avg_xyz_loss)
                    summary1.value.add(tag="axag_loss_per_class_train/" + class_names[i], simple_value=avg_axag_loss)
                    summary1.value.add(tag="trans_loss_per_class_train/" + class_names[i], simple_value=avg_trans_loss)
                    summary1.value.add(tag="sample_count_per_class_train/"+class_names[i], simple_value=count_perClass[i])
                    xyz_loss_perClass[i] = []
                    axag_loss_perClass[i] = []
                    trans_loss_perClass[i] = []

                train_writer.add_summary(summary, step)
                train_writer.add_summary(summary1, step)

            total_seen += batch_size
            batch_idx = batch_idx + 1

        except tf.errors.OutOfRangeError:
            print('End of data!')
            if (epoch+1) % 50 == 0:
                model_dir = "model_" + str(epoch) + ".ckpt"
            else:
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
    general.add_argument('--total_num_point', type=int, default=512, help='Dataset Point Number [256/512/1024] [default: 1024]')

    train_opts = parser.add_argument_group('training_options')
    train_opts.add_argument('--max_epoch', type=int, default=500, help='Epoch to run [default: 100]')
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