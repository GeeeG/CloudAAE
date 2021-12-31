import tensorflow as tf
import numpy as np
import read_yml
import sys


def read_frame_id(filename):
    string = filename.split('/')
    frame_id = int((string[-1].split('.'))[0])
    return frame_id


def read_image(filename, channels, dtype=tf.uint8):
    image_string = tf.read_file(filename)
    return tf.image.decode_png(image_string, channels=channels, dtype=dtype)


def read_files(x, target_cls):
    x['frame_id'] = tf.py_func(read_frame_id, [x['rgb']], tf.int64)
    x['rgb_img'] = read_image(x['rgb'], 0)[:, :, :3]
    x['depth_img'] = read_image(x['depth'], 0, tf.uint16)
    x['mask_img'] = read_image(x['mask'], 1, tf.uint8)
    # need frame_id to read the correct pose info.
    x['cam_intrin'], x['depth_scale'] = tf.py_func(read_yml.read_cam_intrin, [x['info'], x['frame_id'], target_cls], (tf.float64, tf.float64))
    x['translation'], x['rotation_mat'] = tf.py_func(read_yml.read_poses, [x['gt'], x['frame_id'], target_cls], (tf.float64, tf.float64))
    x['fx'], x['fy'], x['cx'], x['cy'] = tf.py_func(read_yml.decompose_cam_intrin, [x['cam_intrin']], (tf.float64, tf.float64, tf.float64, tf.float64))

    if (target_cls == 1 or target_cls == 2):
        x['class_id'] = target_cls - 1
    elif (target_cls == 4 or target_cls == 5 or target_cls == 6):
        x['class_id'] = target_cls - 2
    else:
        x['class_id'] = target_cls - 3 # to zero based class index

    return x