import numpy as np
import scipy as sp
from enum import Enum 
from scipy.spatial.transform import Rotation as R
from quaternion import from_rotation_matrix, quaternion, as_float_array

class State(Enum):
    RESET = 1
    SEARCH_WPTPOKEOUT = 2
    SEARCH_WPTPOKEIN = 3
    CLEAR = 4
    SEARCH_WPTGRASPOUT = 5
    SEARCH_WPTGRASPIN = 6
    GRASP = 7
    PULL = 8
    MOVE_UP = 9
    SEARCH_PLACEMENT = 10
    PLACE = 11
    ALIGN_IN = 12
    ALIGN_OUT = 13
    MOVE_UP2 = 14
    HOME = 15

def skew(x):
    return np.array([[0, -x[2], x[1]],
                    [x[2], 0, -x[0]],
                    [-x[1], x[0], 0]])

def sample_normal_pose(pos_scale, rot_scale):
    '''
    Samples a 6D pose from a zero-mean isotropic normal distribution
    '''
    pos = np.random.normal(scale=pos_scale)
        
    eps = skew(np.random.normal(scale=rot_scale))
    R = sp.linalg.expm(eps)
    quat_wxyz = from_rotation_matrix(R)

    return pos, quat_wxyz

class NoisyObjectPoseSensor:
    def __init__(self, env):
        self._env = env

        self._pos_scale = [0.005] * 3
        self._rot_scale = [0.01] * 3
        # #no noise
        # self._pos_scale = [0] *3
        # self._rot_scale = [0] * 3

    def get_poses(self):
        objs = self._env._scene._active_task.get_base().get_objects_in_tree(exclude_base=True, first_generation_only=False)
        obj_poses = {}

        for obj in objs:
            name = obj.get_name()
            pose = obj.get_pose()

            pos, quat_wxyz = sample_normal_pose(self._pos_scale, self._rot_scale)
            gt_quat_wxyz = quaternion(pose[6], pose[3], pose[4], pose[5])
            perturbed_quat_wxyz = quat_wxyz * gt_quat_wxyz

            pose[:3] += pos
            pose[3:] = [perturbed_quat_wxyz.x, perturbed_quat_wxyz.y, perturbed_quat_wxyz.z, perturbed_quat_wxyz.w]

            obj_poses[name] = pose

        return obj_poses

def euler_to_quat(roll, pitch, yaw):
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    return np.array([qx, qy, qz, qw])

def moving_avg(obj_pose_sensor):
    mov_avg = 50
    obj_poses = obj_pose_sensor.get_poses()
    for _ in range (mov_avg-1):
        for key, item in obj_pose_sensor.get_poses().items():
            obj_poses[key][:3] = [sum(i) for i in zip(obj_poses[key][:3], item[:3])] 
    for key, item in obj_poses.items():
        obj_poses[key][:3] = [i / mov_avg for i in item[:3]] 
    return obj_poses

def rotation_matrix(axis,angle):
    if axis == 'x':
        R = np.array([[1,0,0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
    elif axis == 'y':
        R = np.array([[np.cos(angle),0,np.sin(angle)], [0, 1, 0], [-np.sin(angle),0,np.cos(angle)]])
    elif axis == 'z':
        R = np.array([[np.cos(angle),-np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    return from_rotation_matrix(R)
    
#transform relative to global frame 
def transform(reference_object,dx,dy,dz):
    quat = reference_object[3:7]
    T = np.zeros((4,4))
    T[3, 3] = 1
    T[:3, :3] = R.from_quat(quat).as_matrix()
    w_dx,w_dy,w_dz = (T @ np.array([dx,dy,dz, 0]).reshape((4,1)))[:3, 0]   #vector last element is 0. point is 1
    return w_dx,w_dy,w_dz
