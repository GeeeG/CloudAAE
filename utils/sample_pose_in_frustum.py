import tensorflow as tf
import numpy as np
import sys
sys.path.append('../losses')
import angular_distance_taylor


def sample_rot(npoints, ndim=3):

    # sample on uni sphere http: // mathworld.wolfram.com / SpherePointPicking.html
    theta = tf.random_uniform([1], 0, 2 * np.pi)
    u = tf.random_uniform([1], -1, 1)
    x = tf.sqrt(1 - u * u) * tf.cos(theta)
    y = tf.sqrt(1 - u * u) * tf.sin(theta)
    z = u
    axis = tf.expand_dims(tf.concat([x, y, z], 0), 0)

    # uniformly sample angle
    angle = tf.random.uniform([npoints, 1], minval=-np.pi, maxval=np.pi)

    axag = tf.dtypes.cast(axis * angle, dtype=tf.float64)
    rot_mat = angular_distance_taylor.exponential_map(axag)

    # rot_final_mat = tf.matmul(rot_mat, x_rot_mat)
    rot_final_mat = rot_mat

    return axag, rot_final_mat


def rotation_generation(x):

    # sample an axis-angle
    rot_gen_axag, rot_gen_mat = sample_rot(npoints=1)

    x['rot_gen_mat'] = tf.dtypes.cast(rot_gen_mat, dtype=tf.float32)

    x['rot_gen_axag'] = rot_gen_axag

    return x


def get_frustum(vertical_fov, nearDist, farDist, ratio):
    # http://www.lighthouse3d.com/tutorials/view-frustum-culling/geometric-approach-extracting-the-planes/

    Hnear = 2 * tf.math.tan(vertical_fov / 2) * nearDist
    Wnear = Hnear * ratio

    Hfar = 2 * tf.math.tan(vertical_fov / 2) * farDist
    Wfar = Hfar * ratio

    cam_center = tf.constant([0., 0., 0.])
    cam_direction = tf.constant([0., 0., 1.])
    up_direction = tf.constant([0., 1., 0.])
    right_direction = tf.linalg.cross(up_direction, cam_direction)

    fc = cam_center + cam_direction * farDist
    ftl = fc + (up_direction * Hfar/2) - (right_direction * Wfar/2)
    ftr = fc + (up_direction * Hfar/2) + (right_direction * Wfar/2)
    fbl = fc - (up_direction * Hfar/2) - (right_direction * Wfar/2)
    fbr = fc - (up_direction * Hfar/2) + (right_direction * Wfar/2)

    nc = cam_center + cam_direction * nearDist
    ntl = nc + (up_direction * Hnear/2) - (right_direction * Wnear/2)
    ntr = nc + (up_direction * Hnear/2) + (right_direction * Wnear/2)
    nbl = nc - (up_direction * Hnear/2) - (right_direction * Wnear/2)
    nbr = nc - (up_direction * Hnear/2) + (right_direction * Wnear/2)

    frustum_corners = tf.stack([ftl, ftr, fbl, fbr, ntl, ntr, nbl, nbr], axis=1)

    return frustum_corners, Hnear, Wnear, Hfar, Wfar


def in_frustum_translation(npoints, Wnear, Wfar, farDist, nearDist):

    x = tf.random.normal([npoints, 1], mean=0.0, stddev=(Wnear + Wfar) / 7)
    y = tf.random.normal([npoints, 1], mean=0.0, stddev=(Wnear + Wfar) / 7)
    z = tf.random.normal([npoints, 1], mean=(farDist + nearDist) / 2, stddev=(farDist - nearDist) / 7)
    one = tf.constant([[1.]]) # homogenous coordinate

    frustum_middle = tf.stack([tf.constant([0.]), tf.constant([0.]), (farDist + nearDist) / 2, tf.constant([1.])], axis=0)

    return tf.concat([x, y, z, one], axis=1), tf.transpose(frustum_middle)


def get_proj_matrix(cam_intrin, extrin_rot, extrin_trans):

    cam_extrin = tf.concat([extrin_rot, extrin_trans], axis=1)
    proj_matrix = tf.matmul(cam_intrin, cam_extrin)

    return proj_matrix


def project_pts_to_image(proj_matrix, pts_3d):

    # https://github.com/charlesq34/frustum-pointnets/blob/889c277144a33818ddf73c4665753975f9397fc4/kitti/kitti_util.py
    pts_2d = tf.matmul(proj_matrix, pts_3d)
    pts_2d = tf.transpose(pts_2d)
    pts_2d_x = pts_2d[:, 0] / pts_2d[:, 2]
    pts_2d_y = pts_2d[:, 1] / pts_2d[:, 2]

    return tf.concat([pts_2d_x, pts_2d_y], axis=0)


def check_pts_in_image_fov(pts_2d, xmax, ymax):

    # check x coordinate
    b_x_lower = tf.math.greater(pts_2d[0], tf.constant([0.]))
    b_x_higher = tf.math.less(pts_2d[0], xmax)
    b_x = tf.math.logical_and(b_x_lower, b_x_higher)

    # check y coordinate
    b_y_lower = tf.math.greater(pts_2d[1], tf.constant([0.]))
    b_y_higher = tf.math.less(pts_2d[1], ymax)
    b_y = tf.math.logical_and(b_y_lower, b_y_higher)

    return tf.math.logical_and(b_x, b_y)


def get_final_translation(proj_matrix, pts_3d, xmax, ymax, frustum_middle):

    pts_2d = project_pts_to_image(proj_matrix, tf.transpose(pts_3d))
    b_in_image_fov = check_pts_in_image_fov(pts_2d, xmax, ymax)

    return tf.where(b_in_image_fov, pts_3d, frustum_middle), pts_2d


def translation_generation(x):

    # parameter for linemod dataset camera (primesense)
    vertical_fov = tf.constant([45.])
    nearDist = tf.constant([0.4])
    farDist = tf.constant([1.5])
    ratio = tf.constant([57.5 / 45.])
    cam_intrin = tf.constant([[572.4114, 0.0, 325.2611], [0.0, 573.57043, 242.04899], [0.0, 0.0, 1.0]])
    extrin_rot = tf.eye(3)
    extrin_trans = tf.zeros([3, 1], tf.float32)
    img_height = tf.constant([480.])
    img_width = tf.constant([640.])

    # calculate frustum for visual check
    frustum_corners, Hnear, Wnear, Hfar, Wfar = get_frustum(vertical_fov, nearDist, farDist, ratio)

    # generate translation
    translation_generated, frustum_middle = in_frustum_translation(1, Wnear, Wfar, farDist, nearDist)
    proj_matrix = get_proj_matrix(cam_intrin, extrin_rot, extrin_trans)
    translation_final, _ = get_final_translation(proj_matrix, translation_generated,
                                                 img_width, img_height, frustum_middle)

    x['frustum_corners'] = frustum_corners

    x['trans_gen'] = translation_final[:, 0:3]

    return x
