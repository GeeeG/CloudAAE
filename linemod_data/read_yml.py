import numpy as np
import yaml
import sys


def read_cam_intrin(filename, seq_id, target_cls):
    with open(filename, 'r') as stream:
        try:
            info_files = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    cam_intrin = info_files[seq_id]['cam_K']
    depth_scale = info_files[seq_id]['depth_scale'] * 1000 # convert to meter

    return cam_intrin, depth_scale


def read_poses(filename, frame_id, target_cls):
    print("filename for reading: ", filename)
    print("class index for reading: ", target_cls)

    if target_cls == 2:
        reading_idx = 1
    else:
        reading_idx = 0

    with open(filename, 'r') as stream:
        try:
            pose_files = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

        print("frame id, ", frame_id)
        print("read obj_id ", pose_files[frame_id][reading_idx]['obj_id'])

        if np.equal(target_cls, pose_files[frame_id][reading_idx]['obj_id']):
            translation = [x * 0.001 for x in pose_files[frame_id][reading_idx]['cam_t_m2c']]
            rotation = pose_files[frame_id][reading_idx]['cam_R_m2c']
        elif np.equal(target_cls, pose_files[frame_id][reading_idx-1]['obj_id']): # for frame 993,994
            translation = [x * 0.001 for x in pose_files[frame_id][reading_idx]['cam_t_m2c']]
            rotation = pose_files[frame_id][reading_idx]['cam_R_m2c']
        else:
            print('Object class mismatch!')
            return

        return translation, rotation


def decompose_cam_intrin(cam_intrin):
    fx = np.array(cam_intrin[0])
    fy = np.array(cam_intrin[4])
    cx = np.array(cam_intrin[2])
    cy = np.array(cam_intrin[5])
    return fx, fy, cx, cy