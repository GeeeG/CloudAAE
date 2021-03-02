import tensorflow as tf
import numpy as np
from sample_pose_in_frustum import *

def get_random_object_occluder(x, NUM_CLASS):

    # pick occluder object
    occluder_class_id = np.squeeze((np.random.choice(NUM_CLASS, 1)).astype(np.int32))
    occluder_obj_model = tf.expand_dims(x['obj_model'][occluder_class_id, 0:512,0:3],0)

    # parameter for linemod dataset camera (primesense)
    vertical_fov = tf.constant([45.])
    nearDist = tf.constant([0.4])
    farDist = tf.constant([1.5])
    ratio = tf.constant([57.5 / 45.])

    # calculate frustum for visual check
    frustum_corners, Hnear, Wnear, Hfar, Wfar = get_frustum(vertical_fov, nearDist, farDist, ratio)

    object_trans_z = tf.squeeze(x['translation'])[2]

    num_occluder = 1

    occluder_position_x = tf.random.normal((num_occluder,1), 0, Wnear / 8)
    occluder_position_y = tf.random.normal((num_occluder,1), 0, Hnear / 8)
    occluder_position_z = tf.random.normal((num_occluder,1), (nearDist + object_trans_z) / 2, (object_trans_z - nearDist) / 6)

    occluder_position = tf.stack([occluder_position_x, occluder_position_y, occluder_position_z], axis=1)
    occluder_position = tf.squeeze(occluder_position, axis=2)
    # transform occluder object with pose
    occluder_object_rot = tf.matmul(occluder_obj_model, tf.transpose(x['rot_gen_mat'], perm=[0, 2, 1]))
    x['occluder'] = occluder_object_rot + tf.tile(tf.expand_dims(occluder_position, 1), [1, 512, 1])
    x['frustum_corners'] = frustum_corners

    return x


def get_random_spherical_occluder(x, dataset):

    if(dataset=='linemod'):
        # parameter for linemod dataset camera (primesense)
        vertical_fov = tf.constant([45.])
        nearDist = tf.constant([0.4])
        farDist = tf.constant([1.5])
        ratio = tf.constant([57.5 / 45.])

    if(dataset=='ycbv'):
        vertical_fov = tf.constant([45.])
        nearDist = tf.constant([0.5])
        farDist = tf.constant([1.])
        ratio = tf.constant([58. / 45.])

    # calculate frustum for visual check
    frustum_corners, Hnear, Wnear, Hfar, Wfar = get_frustum(vertical_fov, nearDist, farDist, ratio)

    object_trans_z = tf.squeeze(x['translation'])[2]

    num_occluder = 2

    occluder_position_x = tf.random.normal((num_occluder,1), 0, Wnear / 10)
    occluder_position_y = tf.random.normal((num_occluder,1), 0, Hnear / 10)
    occluder_position_z = tf.random.normal((num_occluder,1), (nearDist + object_trans_z) / 2, (object_trans_z - nearDist) / 6)

    occluder_position = tf.stack([occluder_position_x, occluder_position_y, occluder_position_z], axis=1)
    occluder_position = tf.squeeze(occluder_position)

    sigma = tf.constant([0.01])
    occluder_x_1 = tf.random.normal((200, 1), occluder_position[0,0], sigma)
    occluder_y_1 = tf.random.normal((200, 1), occluder_position[0,1], sigma)
    occluder_z_1 = tf.random.normal((200, 1), occluder_position[0,2], sigma)

    occluder_x_2 = tf.random.normal((200, 1), occluder_position[1,0], sigma)
    occluder_y_2 = tf.random.normal((200, 1), occluder_position[1,1], sigma)
    occluder_z_2 = tf.random.normal((200, 1), occluder_position[1,2], sigma)

    occluder = tf.concat([occluder_x_1, occluder_y_1, occluder_z_1, occluder_x_2, occluder_y_2, occluder_z_2], axis=1)

    x['frustum_corners'] = frustum_corners
    x['occluder'] = tf.reshape(occluder, [1, -1, 3])

    return x
