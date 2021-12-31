import tensorflow as tf
import numpy as np
import os
from collections import defaultdict
import sys
import open3d
import data_gen_tools_linemod
import random

BASE_DIR = '/data_c/PointNet/pointnet/'
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'losses'))
import angular_distance_taylor

b_visual = False

NUM_CLASS = 15
NUM_POINT = 512
# threshold_distance_per_class = 0.05 * np.ones((NUM_CLASS,), dtype=np.float64)
threshold_distance_per_class = np.array([0.05, 0.15, 0.1, 0.1, 0.1, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.1], dtype=np.float64)
# threshold_distance_per_class = [np.asarray(c).astype(np.float64) for c in threshold_distance_per_class]

data_dir = '/data_c/linemod/Linemod_preprocessed/data'
seg_data_dir = '/data_c/linemod/Linemod_preprocessed/segnet_results'
out_dir = '/data_c/linemod/tfRecords_occlu_with_data_source/'
obj_class = range(1, 15)
keynames_path = ['rgb', 'mask', 'depth']
metadata_path = ['gt', 'info']


def get_x_dict(target_cls, data_prefix):
    filepath = os.path.join(data_dir, str('{:02d}'.format(target_cls)))
    filename_tmp = os.path.join(filepath, data_prefix)
    filename = filename_tmp + ".txt"
    x_dict = defaultdict(list)
    for k in keynames_path:
        with open(filename, 'r') as d:
            filelist = d.read().splitlines()
            if (data_prefix == 'test' and k == 'mask'):
                segfilepath = os.path.join(seg_data_dir, str('{:02d}_label'.format(target_cls)))
                x_dict[k] = [segfilepath + "/" + s + "_label.png" for s in filelist]
            else:
                x_dict[k] = [os.path.join(filepath, k) + "/" + s + ".png" for s in filelist]
    for k in metadata_path:
        current_path = os.path.join(filepath, k+".yml")
        repeat_path = np.repeat(current_path, len(x_dict['rgb']), axis=0)
        x_dict[k] = repeat_path
    x_dict = dict(x_dict)
    return x_dict


def get_segment_cloud(x):
    depth_flat = tf.reshape(x['depth_img'], [-1])
    depth_valid_mask = tf.not_equal(tf.cast(x['depth_img'], tf.int16), 0)

    segment_mask = tf.not_equal(x['mask_img'], 0)
    idx = tf.where(tf.logical_and(segment_mask, depth_valid_mask))
    nrows = tf.shape(x['mask_img'], out_type=tf.int64)[0]
    ncols = tf.shape(x['mask_img'], out_type=tf.int64)[1]
    linear_idx = idx[:, 0] * ncols + idx[:, 1]  # fixed!
    depth_segment = tf.gather(depth_flat, linear_idx)

    x['depth_segment'] = depth_segment

    Z = tf.cast(depth_segment, tf.float64) / x['depth_scale']
    Y = (tf.cast(idx[:, 0], tf.float64) - x['cy']) * Z / x['fy']
    X = (tf.cast(idx[:, 1], tf.float64) - x['cx']) * Z / x['fx']
    x['xyz'] = tf.stack([X, Y, Z], axis=1)

    rgb_flat = tf.reshape(x['rgb_img'], [-1, 3])
    rgb_flat_normalize = tf.cast(rgb_flat, tf.float64)/255.
    x['rgb'] = tf.gather(rgb_flat_normalize, linear_idx)

    return x


def segment_mean_distance_filter(xyz, threshold_distance):
    # Filtering based on distance from mean of segment
    segment_average_xyz = tf.reduce_mean(xyz, axis=0)
    d = tf.norm(xyz-segment_average_xyz, ord='euclidean', axis=1)
    return tf.less_equal(d, threshold_distance)


def filter_outlier(x):
    mask_flat = tf.reshape(x['mask_img'], [-1])
    mask_flat_boolean = tf.not_equal(tf.cast(mask_flat, tf.int16), 0)
    depth_flat = tf.reshape(x['depth_img'], [-1])
    depth_flat_boolean = tf.not_equal(tf.cast(depth_flat, tf.int16), 0)
    mask_boolean = tf.logical_and(mask_flat_boolean, depth_flat_boolean)
    x['mask_boolean'] = mask_boolean

    mask_boolean_r = segment_mean_distance_filter(x['xyz'],
                                                  threshold_distance=tf.gather(threshold_distance_per_class, x["class_id"]))
    x["label_mask_r"] = mask_boolean_r
    x["num_valid_points_in_segment"] = tf.count_nonzero(mask_boolean_r)
    return x


def calc_distances(p0, points):
    return ((p0 - points)**2).sum(axis=1)


def FPS_random(pts, K):
    farthest_pts = np.zeros((K, 3))
    farthest_pts_idx = np.zeros(K)
    upper_bound = pts.shape[0] - 1
    first_idx = random.randint(0, upper_bound)
    farthest_pts[0] = pts[first_idx]
    farthest_pts_idx[0] = first_idx
    distances = calc_distances(farthest_pts[0, 0:3], pts[:, 0:3])
    for i in range(1, K):
        farthest_pts[i] = pts[np.argmax(distances)]
        farthest_pts_idx[i] = np.argmax(distances)
        distances = np.minimum(distances, calc_distances(farthest_pts[i, 0:3], pts[:, 0:3]))
    return farthest_pts_idx.astype(np.int64)


def mat_2_quat(x):
    rot_mat = tf.reshape(x['rotation_mat'], (1, 3, 3))
    rot_ss, _ = angular_distance_taylor.logarithm(rot_mat)
    rot_axag = tf.stack([rot_ss[:, 2, 1], rot_ss[:, 0, 2], rot_ss[:, 1, 0]], axis=0)
    x['axisangle'] = rot_axag
    return x


def segment_FPS(x, num_points_per_sample):
    xyz = x['xyz']
    label_mask_r = x['label_mask_r']
    label_mask_r.set_shape([None])
    idx = tf.py_func(FPS_random, [tf.boolean_mask(xyz, label_mask_r), num_points_per_sample], tf.int64)

    y_out = {'class_id': x['class_id'],
             'frame_id': x['frame_id'],
             'rotation_mat': x['rotation_mat'],
             'axisangle': x['axisangle'],
             'translation': x['translation'],
             }

    y_out["num_valid_points_in_segment"] = x["num_valid_points_in_segment"]
    y_out["xyz"] = tf.gather(tf.boolean_mask(x['xyz'], label_mask_r), idx)
    y_out["rgb"] = tf.gather(tf.boolean_mask(x['rgb'], label_mask_r), idx)

    return y_out


def add_data_source(x, data_prefix_id):
    
    x['data_source'] = data_prefix_id
        
    return x


def get_data(x_dict, target_cls, num_points_per_sample, data_prefix_id):

    dataset = tf.data.Dataset.from_tensor_slices(x_dict)
    # dataset = dataset.shuffle(10000)
    dataset = dataset.map(lambda x: data_gen_tools_linemod.read_files(x, target_cls))
    dataset = dataset.map(get_segment_cloud)
    dataset = dataset.map(filter_outlier)
    dataset = dataset.filter(lambda x: tf.greater_equal(x["num_valid_points_in_segment"], NUM_POINT))
    dataset = dataset.map(mat_2_quat)
    dataset = dataset.map(lambda x: segment_FPS(x, num_points_per_sample))
    dataset = dataset.map(lambda x: add_data_source(x, data_prefix_id))

    return dataset


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    # """Wrapper for inserting float features into Example proto."""
    # if not isinstance(value, list):
    #     value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    # """Wrapper for inserting bytes features into Example proto."""
    # if not isinstance(value, list):
    #     value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def create_example(datasample):
    example = tf.train.Example(features=tf.train.Features(
        feature={
            'class_id': _int64_feature(datasample['class_id'].reshape(-1)),
            'frame_id': _int64_feature(datasample['frame_id'].reshape(-1)),
            'rotation_mat': _float_feature(datasample['rotation_mat'].reshape(-1)),
            'axisangle': _float_feature(datasample['axisangle'].reshape(-1)),
            'translation': _float_feature(datasample['translation'].reshape(-1)),
            'num_valid_points_in_segment': _int64_feature(datasample['num_valid_points_in_segment'].reshape(-1)),
            'xyz': _float_feature(datasample['xyz'].reshape(-1)),
            'rgb': _float_feature(datasample['rgb'].reshape(-1)),
            'data_source': _int64_feature(datasample['data_source'].reshape(-1))
        }
    ))
    return example


def dataset_generator(ds, sess):

    tr_iterator = ds.make_one_shot_iterator()
    next_element = tr_iterator.get_next()

    try:
        while True:
            yield sess.run(next_element)

    except tf.errors.OutOfRangeError:
        pass


def creat_records(ds, record_path):

    counter = 0

    with tf.device('/cpu:0'):

        sess = tf.Session()

        with tf.python_io.TFRecordWriter(record_path) as writer:

            generator = dataset_generator(ds, sess)

            for datasample in generator:

                print "counter: ", counter
                print "class index write: ", datasample['class_id']

                if b_visual:
                    cloud = open3d.PointCloud()
                    # xyz_mean = np.mean(datasample['xyz'], axis=0)
                    rot_mat = np.reshape(datasample['rotation_mat'], (3, 3))
                    xyz_remove_rot = np.dot(datasample['xyz'], rot_mat)
                    xyz_remove_trans = xyz_remove_rot - np.dot(rot_mat.T, datasample['translation'])

                    cloud.points = open3d.Vector3dVector(xyz_remove_trans)
                    # cloud.colors = open3d.Vector3dVector(datasample['rgb'])
                    mesh_frame = open3d.create_mesh_coordinate_frame(size=0.1, origin=[0, 0, 0])
                    open3d.draw_geometries([cloud, mesh_frame])
                example = create_example(datasample)
                writer.write(example.SerializeToString())
                counter = counter + 1


def get_data_set(data_prefix, data_prefix_id):
    ds = []
    # for c in obj_class:
    for c in np.arange(1, 16):
        if (c != 3 and c != 7):
            x_train = get_x_dict(c, data_prefix)
            ds_current = get_data(x_train, c, NUM_POINT, data_prefix_id)

            if (c==1 or c==2):
                write_class = c -1
            elif (c==4 or c==5 or c==6):
                write_class = c -2
            else:
                write_class = c-3

            record_path = out_dir + data_prefix + '_files_FPS512_' + str(write_class) + '_adapt_threshold.tfrecords'
            creat_records(ds_current, record_path=record_path)
    return ds


ds = get_data_set("test", 1)
# read_gt_and_intrinsic(1,"train")
