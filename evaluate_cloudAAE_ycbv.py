import argparse
import math
import numpy as np
import tensorflow as tf
import importlib
import os
import sys
import open3d
import transforms3d
import random

# python evaluate_cloudAAE_ycbv.py --trained_model trained_network/20200908-204328/model.ckpt --batch_size 1 --target_cls 0

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
import psutil
from tf_sampling import farthest_point_sample, gather_point
import chamfer_loss

NUM_CLASS = 21
target_cls = np.arange(21)
b_visual = True
b_icp = True

data_dir = 'ycb_video_data_tfRecords'
object_model_dir = "object_model_tfrecord/obj_models.tfrecords"

valid_filenames = []
train_file_lists = []
valid_file_lists = []
threshold_distance_per_class = 0.2 * np.ones((NUM_CLASS,), dtype=np.float32)
sample_factor = 1

valid_seq_id = [[48, 51, 55, 56], # master_chef_can
                [50, 54, 59], # cracker_box
                [49, 51, 54, 55, 58], # sugar_box
                [50, 51, 53, 55, 57, 59], # tomato_soup_can
                [50, 52], # mustard_bottle
                [48, 49, 52, 59], # tuna_fish_can
                [58], # pudding box
                [58], # gelatin box
                [49, 53, 59], # potted_meat_can
                [50, 56], # banana
                [52, 56, 58], # picther_base
                [51, 54, 55, 57], # bleach_cleanser
                [49, 53], # bowl
                [48, 55], # mug
                [50, 54, 56, 59], # drill
                [55], # wood_block
                [51], # scissors
                [57, 59], # large_marker
                [48, 54], # large_clamp
                [48, 57], # extra_large_clamp
                [57]] # foam_brick


def quat2axag_batch(quaternion):
    quaternion = np.reshape(quaternion, (1, 4))
    axag = np.zeros((quaternion.shape[0], 4), dtype=np.float32)
    for k in range(quaternion.shape[0]):
        axag[k, 0:3], axag[k, 3] = transforms3d.quaternions.quat2axangle(quaternion[k, :])
    return axag


def quat2axag_tf(x, BATCH_SIZE):
    current_batch_axag4 = tf.py_func(quat2axag_batch, [x['quaternion']], tf.float32)
    current_batch_axag4 = tf.reshape(current_batch_axag4, [BATCH_SIZE, 4])
    current_batch_axag = tf.expand_dims(current_batch_axag4[:, 3], 1) * current_batch_axag4[:, 0:3]
    x['axisangle'] = current_batch_axag
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

    x['obj_batch'] = tf.expand_dims(tf.gather(obj_model_tf, x['class_id'], axis=0), axis=0)

    return x


def get_rotation_matrix(x):
    x['axisangle'] = tf.dtypes.cast(x['axisangle'], dtype=tf.float64)
    rot_gt_mat = angular_distance_taylor.exponential_map(x['axisangle'])

    x['rot_mat'] = tf.dtypes.cast(rot_gt_mat, dtype=tf.float32)

    return x


def transform_object_model(x):
    translation = tf.reshape(x['translation'], (1, 3))
    model_xyz_rot = tf.matmul(x['obj_batch'][:, :, 0:3], tf.transpose(x['rot_mat'], perm=[0, 2, 1]))
    x['model_xyz_rot_trans'] = model_xyz_rot + tf.cast(tf.tile(tf.expand_dims(translation, 1), [1, 2048, 1]), tf.float32)

    return x


def decode(x):
    features = tf.parse_single_example(
        x,
        features={
            'image': tf.FixedLenFeature((), tf.string),
            'image_shape': tf.FixedLenFeature((3,), tf.int64),
            'depth': tf.FixedLenFeature((), tf.string),
            'depth_shape': tf.FixedLenFeature((2,), tf.int64),
            'label': tf.FixedLenFeature((), tf.string),
            'label_shape': tf.FixedLenFeature((2,), tf.int64),
            'quaternions': tf.FixedLenFeature([NUM_CLASS, 4], tf.float32),
            'translations': tf.FixedLenFeature([NUM_CLASS, 3], tf.float32),
            'class_one_hot': tf.FixedLenFeature([NUM_CLASS], tf.int64),
            'seq_id': tf.FixedLenFeature([], tf.int64),
            'frame_id': tf.FixedLenFeature([], tf.int64),
            'fx': tf.FixedLenFeature([], tf.float32),
            'fy': tf.FixedLenFeature([], tf.float32),
            'cx': tf.FixedLenFeature([], tf.float32),
            'cy': tf.FixedLenFeature([], tf.float32),
            'factor_depth': tf.FixedLenFeature([], tf.float32),
        })

    image_flat = tf.decode_raw(features["image"], out_type=tf.uint8)
    image = tf.reshape(image_flat, shape=features["image_shape"])

    is_four_channel_image = tf.equal(tf.shape(image)[2], 4)
    image = tf.cond(is_four_channel_image, true_fn=lambda: image[:, :, :3], false_fn=lambda: image)

    features['image'] = image

    depth_flat = tf.decode_raw(features["depth"], out_type=tf.uint16)
    features['depth'] = tf.reshape(depth_flat, shape=features["depth_shape"])

    label_flat = tf.decode_raw(features["label"], out_type=tf.uint8)
    features['label'] = tf.reshape(label_flat, shape=features["label_shape"])

    return features


def get_pointcloud(depth, fx, fy, cx, cy, depth_scaling_factor):
    depth_meters = tf.cast(depth, tf.float32) / depth_scaling_factor

    dshape = tf.shape(depth_meters)
    height = dshape[0]
    width = dshape[1]
    xv = tf.cast(tf.range(width), tf.float32)
    yv = tf.cast(tf.range(height), tf.float32)
    X, Y = tf.meshgrid(xv, yv)

    x = ((X - cx) * depth_meters / fx)
    y = ((Y - cy) * depth_meters / fy)
    xyz = tf.stack([x, y, depth_meters], axis=2)  # (height, width, 3)

    return tf.reshape(xyz, [height * width, 3])


def merge_two_dicts(x, y):
    z = x.copy()  # start with x's keys and values
    z.update(y)  # modifies z with y's keys and values & returns None
    return z


def split_samples(x):
    xyz = get_pointcloud(x["depth"], x["fx"], x["fy"], x["cx"], x["cy"], x["factor_depth"])
    rgb = tf.reshape(tf.image.convert_image_dtype(x['image'], dtype=tf.float32), [-1, 3])

    class_idx = tf.where(x["class_one_hot"])
    classes = tf.reshape(class_idx, [-1])
    quaternions = tf.squeeze(tf.gather(x["quaternions"], class_idx))
    translations = tf.squeeze(tf.gather(x["translations"], class_idx))

    depth_flat = tf.cast(tf.reshape(x["depth"], [-1]), tf.int64)
    depth_valid = tf.not_equal(depth_flat, 0)

    data_static = {'xyz': xyz,
                   'rgb': rgb,
                   'depth_valid': depth_valid,
                   'frame_id': x["frame_id"],
                   'seq_id': x["seq_id"],
                   'label': x["label"],
                   }
    d_static = tf.data.Dataset.from_tensors(data_static).repeat()

    data_dynamic = {'class_id': classes,
                    'quaternion': quaternions,
                    'translation': translations
                    }
    d_dynamic = tf.data.Dataset.from_tensor_slices(data_dynamic)

    ds = tf.data.Dataset.zip((d_static, d_dynamic))
    ds = ds.map(lambda y, x: merge_two_dicts(y, x))
    return ds


def segment_mean_distance_filter(xyz, label_mask, threshold_distance):
    # Filtering based on distance from mean of segment
    segment_average_xyz = tf.reduce_mean(tf.boolean_mask(xyz, label_mask), axis=0)
    d = tf.norm(xyz-segment_average_xyz, ord='euclidean', axis=1)
    return tf.logical_and(label_mask, tf.less_equal(d, threshold_distance))


def calc_distances(p0, points):
    return ((p0 - points)**2).sum(axis=1)


def FPS_random(pts, K, seq_id, frame_id, class_id):
    farthest_pts = np.zeros((K, 3))
    farthest_pts_idx = np.zeros(K)
    upper_bound = pts.shape[0] - 1
    if upper_bound==0:
        print "ZERO seq %d frame %d class %d " % (seq_id, frame_id, class_id)
    if pts.shape[0] < K:
        print "seq %d frame %d class %d segmentpont %d" % (seq_id, frame_id, class_id, pts.shape[0])
    sys.stdout.flush()
    first_idx = random.randint(0, upper_bound)
    farthest_pts[0] = pts[first_idx]
    farthest_pts_idx[0] = first_idx
    distances = calc_distances(farthest_pts[0, 0:3], pts[:, 0:3])
    for i in range(1, K):
        farthest_pts[i] = pts[np.argmax(distances)]
        farthest_pts_idx[i] = np.argmax(distances)
        distances = np.minimum(distances, calc_distances(farthest_pts[i, 0:3], pts[:, 0:3]))
    return farthest_pts_idx.astype(np.int64)


def get_outlier_idx(xyz, nb_points, radius, std_ratio):
    # print("xyz"), xyz.shape
    segment_ptCloud = open3d.geometry.PointCloud()
    segment_ptCloud.points = open3d.utility.Vector3dVector(np.squeeze(xyz))
    _, idx = segment_ptCloud.remove_radius_outlier(nb_points, radius)
    # _, idx = segment_ptCloud.remove_statistical_outlier(nb_points, std_ratio)
    if len(idx) < 512:
        idx = np.arange(xyz.shape[0])
    return np.asarray(idx)


# label exist for not presenting object
def segment_not_empty(x):
    label_flat = tf.cast(tf.reshape(x["label"], [-1]), tf.int64) - 1  # To zero-based class indexing!
    label_mask = tf.logical_and(tf.equal(label_flat, x["class_id"]), x["depth_valid"])
    label_mask_r = segment_mean_distance_filter(x['xyz'], label_mask,
                                                threshold_distance=tf.gather(threshold_distance_per_class,
                                                                             x["class_id"]))
    x['xyz_org'] = tf.boolean_mask(x['xyz'], label_mask)
    x["label_mask_r"] = label_mask_r
    x["num_point_after_filter"] = tf.count_nonzero(label_mask_r)
    return x


def outlier_removal(x):
    label_mask_r = x["label_mask_r"]
    x['xyz_org_distance_filtered'] = tf.boolean_mask(x['xyz'], label_mask_r)
    x['rgb_org_distance_filtered'] = tf.boolean_mask(x['rgb'], label_mask_r)
    # outlier removal
    x['inlier_idx'] = tf.py_func(get_outlier_idx, [x['xyz_org_distance_filtered'], 100, 0.02, 0.5], [tf.int64])
    x["num_valid_points_in_segment"] = tf.count_nonzero(x['inlier_idx'])
    return x


def FPS_sample_segment(x, numpoints):
    x['xyz_inlier_full'] = tf.gather(x['xyz_org_distance_filtered'], tf.squeeze(x['inlier_idx']), axis=0)
    x['rgb_inlier_full'] = tf.gather(x['rgb_org_distance_filtered'], tf.squeeze(x['inlier_idx']), axis=0)

    FPS_inlier_idx = tf.py_func(FPS_random, [x['xyz_inlier_full'], numpoints*sample_factor, x['seq_id'], x['frame_id'], x['class_id']], tf.int64)
    FPS_org_idx = tf.py_func(FPS_random, [x['xyz_org_distance_filtered'], numpoints*sample_factor, x['seq_id'], x['frame_id'], x['class_id']], tf.int64)

    y_out = {'class_id': x['class_id'],
             'seq_id': x['seq_id'],
             'frame_id': x['frame_id'],
             'quaternion': x['quaternion'],
             'translation': x['translation'],
             'num_valid_points_in_segment': x['num_valid_points_in_segment'],
             'xyz_inlier_full': x['xyz_inlier_full'],
             'xyz_org_distance_filtered': x['xyz_org_distance_filtered'],
             'xyz_org': x['xyz_org']
             }

    y_out['xyz'] = tf.gather(x['xyz_org_distance_filtered'], FPS_org_idx)
    y_out['rgb'] = tf.gather(x['rgb_org_distance_filtered'], FPS_org_idx)
    y_out["xyz_inlier"] = tf.gather(x['xyz_inlier_full'], FPS_inlier_idx)
    y_out["rgb_inlier"] = tf.gather(x['rgb_inlier_full'], FPS_inlier_idx)

    return y_out


def create_tfrecord_dataset(filename, num_points_per_sample, minimum_points_in_segment,threshold_distance_per_class, target_cls_choosen):

    ncores = psutil.cpu_count()
    ds = tf.data.TFRecordDataset(filename)
    ds = ds.map(decode)
    ds = ds.filter(lambda x: tf.equal(x["class_one_hot"][target_cls[target_cls_choosen]], 1)) # let frame with target class pass
    ds = ds.flat_map(split_samples)
    ds = ds.map(segment_not_empty)
    ds = ds.filter(lambda x: tf.greater(x["num_point_after_filter"], 100))
    ds = ds.filter(lambda x: tf.equal(x["class_id"], target_cls[target_cls_choosen])) # only take target cls segment
    ds = ds.map(outlier_removal)
    ds = ds.map(lambda x: FPS_sample_segment(x, num_points_per_sample))
    ds = ds.filter(lambda x: tf.greater_equal(x["num_valid_points_in_segment"], minimum_points_in_segment))

    ds = ds.map(lambda x: quat2axag_tf(x, 1))
    ds = ds.map(get_object_model)
    ds = ds.map(get_rotation_matrix)
    ds = ds.map(transform_object_model)
    ds = ds.map(lambda x: sphericalFlip_org(x,
                                            tf.reshape(tf.zeros_like(x['translation']), [1, 3]),
                                            tf.tile(tf.constant([[0.8 * math.pi]]), [1, 1])))

    ds = ds.map(hidden_point_removal_org, num_parallel_calls=ncores)

    return ds


def reshape_element(element, batch_size, num_point):
    element['xyz'] = tf.reshape(element['xyz'], [batch_size, num_point*sample_factor, 3])
    element['xyz_inlier'] = tf.reshape(element['xyz_inlier'], [batch_size, -1, 3])
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

    target_cls_choosen = general_opts['target_cls']
    for i in valid_seq_id[target_cls_choosen]:
        filename = str(i).zfill(4) + "_pcnn.tfrecord"
        valid_file_lists.append(filename)

    valid_file_list = [os.path.join(data_dir, file) for file in valid_file_lists]

    BATCH_SIZE = hyperparameters['batch_size']
    NUM_POINT = general_opts['num_point']
    GPU_INDEX = general_opts['gpu']
    MODEL = importlib.import_module(general_opts['model'])  # import network module
    minimum_points_in_segment = NUM_POINT

    tf.set_random_seed(123456789)

    # double check
    BN_INIT_DECAY = 0.5
    BN_DECAY_DECAY_RATE = 0.5
    BN_DECAY_DECAY_STEP = float(40)
    BN_DECAY_CLIP = 0.99

    # threshold distance: for class index i, remove points further away from segment mean than threshold_distance_per_class[i]
    threshold_distance_per_class = 0.2 * np.ones((NUM_CLASS,), dtype=np.float32)

    with tf.Graph().as_default():

        with tf.device('/cpu:0'):

            with tf.name_scope('prepare_data'):
                start_time_seg = datetime.now()
                val_datasets = [create_tfrecord_dataset(f, NUM_POINT, minimum_points_in_segment, threshold_distance_per_class, target_cls_choosen) for f in
                                valid_file_list]
                val_dataset = tf.data.experimental.sample_from_datasets(val_datasets)
                val_dataset = val_dataset.batch(BATCH_SIZE, drop_remainder=False).prefetch(1)
                val_iterator = val_dataset.make_initializable_iterator()

                iter_handle = tf.placeholder(tf.string, shape=[], name='iterator_handle')
                iterator = tf.data.Iterator.from_string_handle(iter_handle, val_dataset.output_types,
                                                               val_dataset.output_shapes)
                next_element = iterator.get_next()
                next_element = reshape_element(next_element, batch_size=BATCH_SIZE, num_point=NUM_POINT)
                time_elapsed_seg = datetime.now() - start_time_seg
                print 'seg time elapsed (hh:mm:ss) {}'.format(time_elapsed_seg)

        with tf.device('/gpu:' + str(GPU_INDEX)):

            is_training_pl_encoder = tf.placeholder(tf.bool, shape=())
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

            next_element_xyz_inlier = next_element['xyz_inlier'][:, 0:NUM_POINT, :]
            next_element_xyz_inlier = tf.reshape(next_element_xyz_inlier, [BATCH_SIZE, NUM_POINT, 3])

            cls_gt_onehot = tf.one_hot(indices=next_element['class_id'], depth=len(target_cls))
            cls_gt_onehot_expand = tf.expand_dims(cls_gt_onehot, axis=1)
            cls_gt_onehot_tile = tf.tile(cls_gt_onehot_expand, [1, NUM_POINT, 1])

            visiblePoints = tf.reshape(next_element['visiblePoints_org'], [BATCH_SIZE, 2048 + 1, 3])

            visiblePoints_final = visiblePoints[:, 0:NUM_POINT, :]

            xyz_graph_input = next_element_xyz_inlier

            with tf.name_scope('6d_pose'):
                element_mean = tf.reduce_mean(xyz_graph_input, axis=1)

                next_element_xyz_inlier_normalized = xyz_graph_input - tf.expand_dims(element_mean, 1)

                # dgcnn
                xyz_recon_res, rot_pred, trans_pred_res, endpoint = MODEL.get_model_dgcnn_mean_6d(tf.concat([next_element_xyz_inlier_normalized,
                                                                 cls_gt_onehot_tile], axis=2),
                                                                is_training_pl_encoder, is_training_pl, 10, bn_decay=bn_decay)

                xyz_recon = xyz_recon_res + tf.tile(tf.expand_dims(element_mean, 1), [1, xyz_recon_res.shape[1], 1])
                trans_pred = trans_pred_res + element_mean

                # for all decoder
                xyz_recon_FPS = gather_point(xyz_recon, farthest_point_sample(NUM_POINT, xyz_recon))

                xyz_loss, _ = chamfer_loss.get_loss(xyz_recon_FPS, visiblePoints_final)
                tf.summary.scalar('chamfer_loss', xyz_loss)

            with tf.name_scope('translation'):
                trans_loss, trans_loss_perSample = trans_distance.get_translation_error(trans_pred,
                                                                                        next_element['translation'])
                mean_dist_loss, mean_dist_loss_perSample = trans_distance.get_translation_error(element_mean,
                                                                                                next_element[
                                                                                                    'translation'])
                tf.summary.scalar('trans_loss', trans_loss)
                tf.summary.scalar('mean_dist_loss', mean_dist_loss)
                tf.summary.scalar('trans_loss_min', tf.reduce_min(trans_loss_perSample))
                tf.summary.scalar('trans_loss_max', tf.reduce_max(trans_loss_perSample))

            xyz_remove_trans = next_element['xyz'] - tf.expand_dims(trans_pred, axis=1)
            with tf.name_scope('rotation'):
                current_batch_axag = tf.reshape(next_element['axisangle'], (1, 3))

                rot_pred = tf.cast(rot_pred, tf.float64)
                axag_loss, axag_loss_perSample = angular_distance_taylor.get_rotation_error(rot_pred,
                                                                                            tf.cast(current_batch_axag,
                                                                                                    tf.float64))
                axag_loss = tf.cast(axag_loss, tf.float32)
                tf.summary.scalar('axag_loss', axag_loss)
                tf.summary.scalar('axag_loss_min', tf.reduce_min(axag_loss_perSample))
                tf.summary.scalar('axag_loss_max', tf.reduce_max(axag_loss_perSample))

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()

        # Init variables
        init = tf.global_variables_initializer()

        sess.run(init, {is_training_pl_encoder: False, is_training_pl: False})

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver(max_to_keep=None)

        # Restore variables from disk.
        trained_model = general_opts['trained_model']
        saver.restore(sess, trained_model)
        print "Model restored."

        ops = {'is_training_pl': is_training_pl,
               'is_training_pl_encoder': is_training_pl_encoder,
               'trans_loss': trans_loss,
               'axag_loss': axag_loss,
               'merged': merged,
               'step': batch,
               'class_id': next_element['class_id'],
               'seq_id': next_element['seq_id'],
               'frame_id': next_element['frame_id'],
               'obj_batch': tf.squeeze(next_element['obj_batch'], axis=1),
               'trans_pred': trans_pred,
               'rot_pred': rot_pred,
               'xyz_recon': xyz_recon_FPS,
               'xyz_inlier_full': next_element['xyz_inlier_full'],
               'xyz_graph_input': xyz_graph_input,
               'handle': iter_handle}

        validation_handle = sess.run(val_iterator.string_handle())
        sess.run(val_iterator.initializer)

        model, class_id = eval_graph(sess, ops, validation_handle, batch_size=BATCH_SIZE)


def eval_graph(sess, ops, validation_handle, batch_size):
    """ ops: dict mapping from string to tf ops """
    is_training_encoder = False
    is_training = False

    batch_idx = 0
    total_seen = 0
    total_trans_loss = 0.
    total_axag_loss = 0.

    start_time_eval = datetime.now()
    while True:
        try:

            feed_dict = {ops['is_training_pl']: is_training,
                         ops['is_training_pl_encoder']: is_training_encoder,
                         ops['handle']: validation_handle}

            _, step, trans_loss_val, axag_loss_val, \
            class_id, seq_id, frame_id, trans_pred, rot_pred, obj_batch, \
            xyz_recon, xyz_inlier_full, xyz_graph_input = \
                sess.run([ops['merged'],
                          ops['step'],
                          ops['trans_loss'],
                          ops['axag_loss'],
                          ops['class_id'],
                          ops['seq_id'],
                          ops['frame_id'],
                          ops['trans_pred'],
                          ops['rot_pred'],
                          ops['obj_batch'],
                          ops['xyz_recon'],
                          ops['xyz_inlier_full'],
                          ops['xyz_graph_input'],
                          ],
                         feed_dict=feed_dict)

            print "class %d, sequence %d, frame %d" % (class_id, seq_id, frame_id)

            total_seen += batch_size
            total_axag_loss += axag_loss_val
            total_trans_loss += trans_loss_val

            print 'Validation batch %d seq_id %d frame_id %d trans_loss %f rot_loss %f' % (batch_idx, seq_id, frame_id, trans_loss_val, axag_loss_val)
            batch_idx = batch_idx + 1

            if b_visual:
                batch_sample_idx = 0
                current_rot = rot_pred[batch_sample_idx]
                current_ag = np.linalg.norm(current_rot, ord=2)
                current_ax = current_rot / current_ag
                rotmat = transforms3d.axangles.axangle2mat(current_ax, current_ag)
                xyz_remove_rot = np.dot(xyz_inlier_full[batch_sample_idx,:,:], rotmat)
                xyz_remove_trans = xyz_remove_rot - np.dot(rotmat.T, trans_pred[batch_sample_idx,:])

                xyz_recon_remove_rot = np.dot(xyz_recon[batch_sample_idx, :, :], rotmat)
                xyz_recon_remove_trans = xyz_recon_remove_rot - np.dot(rotmat.T, trans_pred[batch_sample_idx, :])

                xyz_graph_input_pCloud = open3d.geometry.PointCloud()
                xyz_graph_input_pCloud.points = open3d.utility.Vector3dVector(xyz_graph_input[batch_sample_idx, :, :])

                segment_ptCloud = open3d.geometry.PointCloud()
                segment_ptCloud.points = open3d.utility.Vector3dVector(xyz_remove_trans)
                segment_ptCloud.paint_uniform_color([0.9, 0.1, 0.1])

                xyz_recon_ptCloud = open3d.geometry.PointCloud()
                xyz_recon_ptCloud.points = open3d.utility.Vector3dVector(xyz_recon_remove_trans)
                xyz_recon_ptCloud.paint_uniform_color([0.1, 0.1, 0.9])

                model_pCloud = open3d.geometry.PointCloud()
                model_pCloud.points = open3d.utility.Vector3dVector(obj_batch[batch_sample_idx, :, 0:3])
                model_pCloud.colors = open3d.utility.Vector3dVector(obj_batch[batch_sample_idx, :, 3:6])
                model_pCloud.paint_uniform_color([0.1, 0.9, 0.1])

                model_frame = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

                print('Visualization before ICP refinement')
                open3d.visualization.draw_geometries([model_pCloud, segment_ptCloud, model_frame])
                open3d.visualization.draw_geometries([segment_ptCloud, xyz_recon_ptCloud])

                # ICP
                if b_icp:

                    trans = trans_pred[batch_sample_idx, :]
                    trans.shape = (3, 1)
                    segment_transform = np.vstack([np.hstack([rotmat, trans]), np.array([0., 0., 0., 1.])])

                    radius = 0.01
                    evaluation = open3d.registration.evaluate_registration(model_pCloud, xyz_graph_input_pCloud,
                                                                           radius, segment_transform)

                    for i in np.arange(0, 10):

                        reg_p2p = open3d.registration.registration_icp(model_pCloud, xyz_graph_input_pCloud, radius, segment_transform,
                                                                       open3d.registration.TransformationEstimationPointToPoint())
                        radius = radius * 0.9

                        segment_transform = reg_p2p.transformation

                        rotmat_icp = segment_transform[:3, :3]
                        translation_icp = segment_transform[0:3, 3]

                    xyz_remove_rot_w_icp = np.dot(xyz_inlier_full[batch_sample_idx, :, :], rotmat_icp)
                    xyz_remove_trans_w_icp = xyz_remove_rot_w_icp - np.dot(rotmat_icp.T, translation_icp)
                    pred_recon_pCloud_after_icp = open3d.geometry.PointCloud()
                    pred_recon_pCloud_after_icp.points = open3d.utility.Vector3dVector(xyz_remove_trans_w_icp)
                    pred_recon_pCloud_after_icp.paint_uniform_color([0.9, 0.1, 0.1])

                    xyz_recon_remove_rot_w_icp = np.dot(xyz_recon[batch_sample_idx, :, :], rotmat_icp)
                    xyz_recon_remove_trans_w_icp = xyz_recon_remove_rot_w_icp - np.dot(rotmat_icp.T, translation_icp)
                    xyz_recon_ptCloud_after_icp = open3d.geometry.PointCloud()
                    xyz_recon_ptCloud_after_icp.points = open3d.utility.Vector3dVector(xyz_recon_remove_trans_w_icp)
                    xyz_recon_ptCloud_after_icp.paint_uniform_color([0.1, 0.1, 0.9])

                    print('Visualization after ICP refinement')

                    open3d.visualization.draw_geometries([model_pCloud, pred_recon_pCloud_after_icp, model_frame])
                    open3d.visualization.draw_geometries([pred_recon_pCloud_after_icp, xyz_recon_ptCloud_after_icp])


        except tf.errors.OutOfRangeError:
            print('End of data!')
            break

    avg_axag_loss = total_axag_loss / float(batch_idx)
    avg_trans_loss = total_trans_loss / float(batch_idx)

    time_elapsed = datetime.now() - start_time_eval
    print "eval time elapsed (hh:mm:ss.ms) {}".format(time_elapsed)
    print "batch size %d" % batch_idx
    print "trans_loss %f axag_loss %f" \
          % (avg_trans_loss, avg_axag_loss)

    return obj_batch[batch_sample_idx, :, :], class_id[batch_sample_idx]


def get_training_argparser():
    parser = argparse.ArgumentParser()

    general = parser.add_argument_group('general')
    general.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
    general.add_argument('--model', default='pointnet_ycb_23_decoder_4', help='Model name')
    general.add_argument('--log_dir', default='log', help='Log dir [default: log]')
    general.add_argument('--num_point', type=int, default=256, help='Point Number [256/512/1024/2048] [default: 256]')
    general.add_argument('--target_cls', type=int, default=9, help='Target testing class [default:14]')
    general.add_argument('--trained_model', help='Absolute path to trained model')

    train_opts = parser.add_argument_group('training_options')
    train_opts.add_argument('--max_epoch', type=int, default=300, help='Epoch to run [default: 100]')
    train_opts.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')

    hyperparameters = parser.add_argument_group('hyperparameters')
    hyperparameters.add_argument('--batch_size', type=int, default=128,
                                 help='Batch Size during training [default: 128]')
    hyperparameters.add_argument('--learning_rate', type=float, default=0.008,
                                 help='Initial learning rate [default: 0.008]')
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