import tensorflow as tf
import numpy as np
import scipy.io
from scipy.spatial import ConvexHull

def sphericalFlip(x, center, param):
    points = tf.concat([x['model_xyz_rot_trans'], x['occluder']], 1)
    # points = tf.concat([x['model_xyz_rot_trans'], x['seg_noise']], 1)
    # points = x['model_xyz_rot_trans']
    num_points = points.shape[1]
    points = points - tf.tile(tf.expand_dims(center, 1), [1, num_points, 1])
    normPoints = tf.norm(points, ord='euclidean', axis=2)
    R = tf.tile((tf.math.reduce_max(normPoints, axis=1, keepdims=True) * tf.math.pow(10.0, param)),
                [1, num_points])
    flippedPointsTemp = 2 * tf.tile(tf.expand_dims(R - normPoints, 2), [1, 1, points.shape[2]]) * points
    flippedPoints = tf.math.divide(flippedPointsTemp,  tf.tile(tf.expand_dims(normPoints, 2), [1, 1, points.shape[2]]))
    flippedPoints += points

    x['flippedPoints'] = tf.concat([flippedPoints, tf.zeros([flippedPoints.shape[0], 1, flippedPoints.shape[2]])],
                                   axis=1)
    x['orgPoints'] = tf.concat([points, tf.zeros([flippedPoints.shape[0], 1, flippedPoints.shape[2]])],
                               axis=1)

    return x


def convexHull(points, orgPoints):
    visiblePoints = np.zeros((points.shape[0], points.shape[1], points.shape[2]), dtype=np.float32)
    num_vis_point = np.zeros((points.shape[0]), dtype=np.int64)
    for k in range(points.shape[0]):
        flag = np.zeros(points.shape[1], int)
        hull = ConvexHull(points[k, ...])
        visibleVertex = hull.vertices[:-1]
        flag[visibleVertex] = 1
        visibleId = np.where(flag == 1)[0]
        visibleId = visibleId[:-1]  # remove idex for 0.

        random_idx = np.random.choice(visibleId, points.shape[1] - len(visibleId))

        visiblePoints[k, ...] = orgPoints[k, np.concatenate((visibleId, random_idx), axis=0), :]
        num_vis_point[k] = len(visibleId)

    return visiblePoints, num_vis_point


def hidden_point_removal(x):
    x['visiblePoints'], x['num_vis_point'] = tf.py_func(convexHull, [x['flippedPoints'], x['orgPoints']], [tf.float32, tf.int64])
    return x


def sphericalFlip_org(x, center, param):
    # points = tf.concat([x['model_xyz_rot_trans'], x['seg_noise']], 1)
    points = x['model_xyz_rot_trans']
    num_points = points.shape[1]
    points = points - tf.tile(tf.expand_dims(center, 1), [1, num_points, 1])
    normPoints = tf.norm(points, ord='euclidean', axis=2)
    R = tf.tile((tf.math.reduce_max(normPoints, axis=1, keepdims=True) * tf.math.pow(10.0, param)),
                [1, num_points])
    flippedPointsTemp = 2 * tf.tile(tf.expand_dims(R - normPoints, 2), [1, 1, points.shape[2]]) * points
    flippedPoints = tf.math.divide(flippedPointsTemp,  tf.tile(tf.expand_dims(normPoints, 2), [1, 1, points.shape[2]]))
    flippedPoints += points

    x['flippedPoints_org'] = tf.concat([flippedPoints, tf.zeros([flippedPoints.shape[0], 1, flippedPoints.shape[2]])],
                                   axis=1)
    x['orgPoints_org'] = tf.concat([points, tf.zeros([flippedPoints.shape[0], 1, flippedPoints.shape[2]])],
                               axis=1)

    return x


def hidden_point_removal_org(x):
    x['visiblePoints_org'], x['num_vis_point_org'] = tf.py_func(convexHull, [x['flippedPoints_org'], x['orgPoints_org']], [tf.float32, tf.int64])
    return x


def sphericalFlip_real(x, center, param):
    points = tf.concat([x['xyz'], x['occluder']], 1)
    # points = tf.concat([x['model_xyz_rot_trans'], x['seg_noise']], 1)
    # points = x['model_xyz_rot_trans']
    num_points = points.shape[1]
    points = points - tf.tile(tf.expand_dims(center, 1), [1, num_points, 1])
    normPoints = tf.norm(points, ord='euclidean', axis=2)
    R = tf.tile((tf.math.reduce_max(normPoints, axis=1, keepdims=True) * tf.math.pow(10.0, param)),
                [1, num_points])
    flippedPointsTemp = 2 * tf.tile(tf.expand_dims(R - normPoints, 2), [1, 1, points.shape[2]]) * points
    flippedPoints = tf.math.divide(flippedPointsTemp,  tf.tile(tf.expand_dims(normPoints, 2), [1, 1, points.shape[2]]))
    flippedPoints += points

    x['flippedPoints_real'] = tf.concat([flippedPoints, tf.zeros([flippedPoints.shape[0], 1, flippedPoints.shape[2]])],
                                   axis=1)
    x['orgPoints_real'] = tf.concat([points, tf.zeros([flippedPoints.shape[0], 1, flippedPoints.shape[2]])],
                               axis=1)

    return x


def hidden_point_removal_real(x):
    x['visiblePoints_real'], x['num_vis_point_real'] = tf.py_func(convexHull, [x['flippedPoints_real'], x['orgPoints_real']], [tf.float32, tf.int64])
    return x