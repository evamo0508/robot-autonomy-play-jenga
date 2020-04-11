import numpy as np
import scipy as sp
from quaternion import from_rotation_matrix, quaternion

from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import *

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


class RandomAgent:

    # def act(self, obs):
    #     delta_pos = [(np.random.rand() * 2 - 1) * 0.005, 0, 0]
    #     delta_quat = [0, 0, 0, 1] # xyzw
    #     gripper_pos = [np.random.rand() > 0.5]
    #     return delta_pos + delta_quat + gripper_pos

    def act(self, obs, obj_poses,visited):
        #move to waypoint2, then move to waypoint0. visited declared in main
        thresh = 0.05
        gripper_pos = [True]
        waypoint2 = obj_poses['waypoint2'].tolist()
        waypoint1 = obj_poses['waypoint1'].tolist()
        gripper = obs.gripper_pose.tolist()
        print(np.linalg.norm(obs.gripper_pose[:3]-obj_poses['waypoint2'][:3]))
        if visited[0] == False:
            if np.linalg.norm(obs.gripper_pose[:3]-obj_poses['waypoint2'][:3]) > thresh:
                return waypoint2 + gripper_pos, visited
            if np.linalg.norm(obs.gripper_pose[:3]-obj_poses['waypoint2'][:3]) <= thresh:
                visited[0] = True
        if visited[0] == True and visited[1] == False:
            if np.linalg.norm(obs.gripper_pose[:3]-obj_poses['waypoint1'][:3]) > thresh:
                return waypoint1 + gripper_pos, visited
            if np.linalg.norm(obs.gripper_pose[:3]-obj_poses['waypoint1'][:3]) <= thresh:
                visited[1] = True
        if visited[1] == True:
            gripper_pos = [False]
        if gripper_pos == [False]:
            gripper_pos = [False]
            return waypoint2 + gripper_pos, visited
        return gripper + gripper_pos, visited

        # # print("Cuboid1\n",obj_poses['Cuboid1'])
        # # print ("obs gripper pose\n", obs.gripper_pose)
        # print ("obs gripper pose angle?\n",obs.gripper_pose[3:7])
        # print("waypoint1\n",obj_poses['waypoint1'])

        # # move xyz position to block
        # delta_pos = (obj_poses['waypoint1'][:3]-obs.gripper_pose[:3])
        # max_step = 0.5
        # min_step = 0.05
        # delta_pos[delta_pos > max_step] = max_step
        # delta_pos[delta_pos < -max_step] = -max_step
        # delta_pos = np.where(np.abs(delta_pos) < min_step, 0, delta_pos)
        # delta_pos = delta_pos.tolist()
        # print("delta_pos",delta_pos)

        # delta_quat = [0, 0, 0, 1] # xyzw

        # # # move orientation to block
        # # delta_quat = (obj_poses['waypoint1'][3:7]-obs.gripper_pose[3:7])
        # # max_step = 0.02
        # # min_step = 0.0001
        # # delta_quat[delta_quat > max_step] = max_step
        # # delta_quat[delta_quat < -max_step] = -max_step
        # # delta_quat = np.where(np.abs(delta_quat) < min_step, 0, delta_quat)
        # # delta_quat = delta_quat.tolist()
        # # print("delta_quat",delta_quat)

        
        # return delta_pos + delta_quat + gripper_pos


class NoisyObjectPoseSensor:

    def __init__(self, env):
        self._env = env

        self._pos_scale = [0.005] * 3
        self._rot_scale = [0.01] * 3

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


if __name__ == "__main__":
    # action_mode = ActionMode(ArmActionMode.DELTA_EE_POSE) # See rlbench/action_modes.py for other action modes
    # action_mode = ActionMode(ArmActionMode.DELTA_EE_POSE_PLAN)
    action_mode = ActionMode(ArmActionMode.ABS_EE_POSE_PLAN)
    env = Environment(action_mode, '', ObservationConfig(), False)
    # task = env.get_task(StackBlocks) # available tasks: EmptyContainer, PlayJenga, PutGroceriesInCupboard, SetTheTable
    task = env.get_task(PlayJenga) # available tasks: EmptyContainer, PlayJenga, PutGroceriesInCupboard, SetTheTable

    agent = RandomAgent()
    obj_pose_sensor = NoisyObjectPoseSensor(env)
   
    descriptions, obs = task.reset()
    
    
    visited = [False, False,False]
    while True:
        # Getting noisy object poses
        obj_poses = obj_pose_sensor.get_poses()


        # Getting various fields from obs
        current_joints = obs.joint_positions
        gripper_pose = obs.gripper_pose
        rgb = obs.wrist_rgb
        depth = obs.wrist_depth
        mask = obs.wrist_mask

        # Perform action and step simulation
        # action = agent.act(obs)

        print("visited",visited)
        action,visited = agent.act(obs,obj_poses,visited)
        obs, reward, terminate = task.step(action)
        

        # if terminate:
        #     break

    env.shutdown()
