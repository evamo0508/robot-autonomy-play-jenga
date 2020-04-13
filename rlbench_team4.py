import numpy as np
import scipy as sp
from scipy.spatial.transform import Rotation as R
from quaternion import from_rotation_matrix, quaternion, as_float_array

from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import *

from enum import Enum 

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

class State(Enum):
    RESET = 1
    SEARCH_WPTPOKEOUT = 2
    SEARCH_WPTPOKEIN = 3
    CLEAR = 5
    SEARCH_WPTGRASPOUT = 6
    SEARCH_WPTGRASPIN = 6
    GRASP = 7
    PULL = 8
    SEARCH_PLACEMENT = 9
    PLACE = 10
    ALIGN = 11

class Agent:

    def __init__(self, obs, obj_poses):
        self.state = State.RESET
        self.goal = obs.gripper_pose.tolist() # x, y , z, quaternion
        self.gripper = [True]
        self.cuboidX = 0.2067
        self.cuboidY = 0.069
        self.cuboidZ = 0.0333
        self.wptPokeOut = obj_poses['waypoint2']
        self.wptPokeIn =  obj_poses['waypoint1']
        self.wptGraspOut = np.zeros(7)
        self.wptGraspIn = np.zeros(7)
        self.visited = []
        

    def act(self, obs, obj_poses):
        if self.state == State.RESET:
            self.reset()
        elif self.state == State.SEARCH_WPTPOKEOUT:
            self.search_wptPokeOut(obs, obj_poses)
        """
        elif self.state == State.SEARCH_WPTPOKEIN:
            self.search_wptPokeIn(obs, obj_poses)
        elif self.state == State.CLEAR:
            self.clear(obs, obj_poses)
        elif self.state == State.SEARCH_WPTGRASPOUT:
            self.search_wptGraspOut(obs, obj_poses)
        elif self.state == State.SEARCH_WPTGRASPIN:
            self.search_wptGraspIn(obs, obj_poses)
        elif self.state == State.GRASP:
            self.grasp()
        elif self.state == State.PULL:
            self.pull(obs, obj_poses)
        elif self.state == State.SEARCH_PLACEMENT:
            self.search_placement(obs, obj_poses)
        """
        
        #self.state = self.state + 1 if self.state != State.ALIGN else self.RESET
        
        return self.goal + self.gripper

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
    def reset(self):
        self.gripper = [True]
        
        while True:
            cuboidID = np.random.choice(12)
            if cuboidID not in self.visited:
                self.visited.append(cuboidID)
                break
        
        if cuboidID == 10:
            cuboid = 'Cuboid'
        elif cuboidID == 11:
            cuboid = 'target_cuboid'
        else:
            cuboid = 'Cuboid' + str(cuboidID)
        print("Cuboid: ", cuboid) 
        quat = obj_poses[cuboid][3:7]
        T = np.zeros((4,4))
        T[3, 3] = 1
        T[:3, :3] = R.from_quat(quat).as_matrix().T
        T[0, 3] = int(self.cuboidX / 2) + 0.01 # x
        
        y90 = from_rotation_matrix(np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]))
        
        self.wptPokeOut[:3] = (T @ np.append(obj_poses[cuboid][:3], 1))[:3]
        self.wptPokeOut[3:7] = as_float_array(y90 * quaternion(quat[0], quat[1], quat[2], quat[3]))
        print(self.wptPokeOut)

        T[0, 3] = int(self.cuboidX / 2) - 0.03 # x
        self.wptPokeIn[:3] = (T @ np.append(obj_poses[cuboid][:3], 1))[:3]
        self.wptPokeIn[3:7] = as_float_array(y90 * quaternion(quat[0], quat[1], quat[2], quat[3]))
        
        T[0, 3] = - int(self.cuboidX) * 0.025 # x
        self.wptGraspOut[:3] = (T @ np.append(obj_poses[cuboid][:3], 1))[:3]
        self.wptGraspOut[3:7] = as_float_array(y90 * quaternion(quat[0], quat[1], quat[2], quat[3]))
        
        T[0, 3] = - int(self.cuboidX / 2) - 0.015 # x
        self.wptGraspIn[:3] = (T @ np.append(obj_poses[cuboid][:3], 1))[:3]
        self.wptGraspIn[3:7] = as_float_array(y90 * quaternion(quat[0], quat[1], quat[2], quat[3]))

        self.state = State.SEARCH_WPTPOKEOUT
        
    def search_wptPokeOut(self, obs, obj_poses):
        thresh = 0.05
        # TODO: quaternion error
        if np.linalg.norm(obs.gripper_pose[:3] - self.wptPokeOut[:3]) > thresh:
        #if np.linalg.norm(obs.gripper_pose[:3] - obj_poses['waypoint2'][:3]) > thresh:
            self.goal = self.wptPokeOut.tolist()
            #self.goal = obj_poses['waypoint2'].tolist()
        else:
            self.state = State.SEARCH_WPTPOKEIN

    def search_wptPokeIn(self, obs, obj_poses):    
        thresh = 0.05
        # TODO: quaternion error
        if np.linalg.norm(obs.gripper_pose[:3] - self.wptPokeIn[:3]) > thresh:
            self.goal = self.wptPokeIn.tolist()
        else:
            self.state = State.CLEAR
    
    def clear(self, obs, obj_poses):
        thresh = 0.05
        # TODO: quaternion error
        if np.linalg.norm(obs.gripper_pose[:3] - self.wptPokeOut[:3]) > thresh:
            self.goal = self.wptPokeOut.tolist()
        else:
            self.state = State.SEARCH_wptGraspOut
    
    def search_wptGraspOut(self, obs, obj_poses):
        thresh = 0.05
        # TODO: quaternion error
        if np.linalg.norm(obs.gripper_pose[:3] - self.wptGraspOut[:3]) > thresh:
            self.goal = self.wptGraspOut.tolist()
        else:
            self.state = State.SEARCH_wptGraspIn
        
    def search_wptGraspOut(self, obs, obj_poses):
        thresh = 0.05
        # TODO: quaternion error
        if np.linalg.norm(obs.gripper_pose[:3] - self.wptGraspOut[:3]) > thresh:
            self.goal = self.wptGraspOut.tolist()
        else:
            self.state = State.SEARCH_wptGraspIn

    def grasp(self):
        self.gripper = [False]
        self.state = State.PULL
    
    def pull(self, obs, obj_poses):
        thresh = 0.05
        # TODO: quaternion error
        if np.linalg.norm(obs.gripper_pose[:3] - self.wptGraspOut[:3]) > thresh:
            self.goal = self.wptGraspOut.tolist()
        else:
            self.state = State.SEARCH_PLACEMENT

    def search_placement(self, obs, obj_poses):
        pass
        
            
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

    obj_pose_sensor = NoisyObjectPoseSensor(env)
    descriptions, obs = task.reset()
    agent = Agent(obs, obj_pose_sensor.get_poses())
    
    
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

        action = agent.act(obs, obj_poses)
        obs, reward, terminate = task.step(action)
        

        # if terminate:
        #     break

    env.shutdown()
