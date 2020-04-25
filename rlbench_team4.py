import numpy as np
import scipy as sp
from scipy.spatial.transform import Rotation as R
from quaternion import from_rotation_matrix, quaternion, as_float_array

from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import *
from pyrep.objects.shape import Shape

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

class Agent:

    def __init__(self, obs, obj_poses, task):
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
        self.wptPull = []
        self.wptAlignout = []
        self.visited = []
        self.home_goal = self.goal
        self.placePose = []
        self.rdx = 0
        self._jenga = task

        
    def act(self, obs, obj_poses):
        print(self.state)

        if self.state == State.RESET:
            self.reset()
        elif self.state == State.SEARCH_WPTPOKEOUT:
            self.search_wptPokeOut(obs, obj_poses)
        elif self.state == State.SEARCH_WPTPOKEIN:
            self.search_wptPokeIn(obs, obj_poses)
        elif self.state == State.CLEAR:
            self.clear(obs, obj_poses)
        elif self.state == State.SEARCH_WPTGRASPOUT:
            self.search_wptGraspOut(obs, obj_poses)
        elif self.state == State.HOME:
            self.home(obs)
        elif self.state == State.SEARCH_WPTGRASPIN:
            self.search_wptGraspIn(obs, obj_poses)
        elif self.state == State.GRASP:
            self.grasp()
        elif self.state == State.PULL:
            self.pull(obs, obj_poses)
        elif self.state == State.MOVE_UP:
            self.move_up(obs, obj_poses)
        elif self.state == State.SEARCH_PLACEMENT:
            self.search_placement(obs, obj_poses)
        elif self.state == State.PLACE:
            self.place()
        elif self.state == State.ALIGN_IN:
            self.align_in()
        elif self.state == State.ALIGN_OUT:
            self.align_out()
        elif self.state == State.MOVE_UP2:
            self.move_up2(obs, obj_poses)    

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
            cuboidID = np.random.choice(range(5,11))    #dont take bottom level
            if cuboidID not in self.visited:
                self.visited.append(cuboidID)
                break
        

        if cuboidID == 10:
            self.cuboid = 'target_cuboid'
        else:
            self.cuboid = 'Cuboid' + str(cuboidID)

        # test
        # self.cuboid = 'target_cuboid'
        self.cuboid = 'Cuboid3'
        obj_poses['target_cuboid'] = obj_poses['Cuboid3']
        print("Cuboid: ", self.cuboid) 

        self._jenga._task.register_graspable_objects([Shape(self.cuboid)])

        if self.cuboid == "target_cuboid":
            poke_amount = 0.065      #increase means poke more into the block
        else:
            poke_amount = 0.045
        quat = obj_poses[self.cuboid][3:7]
        T = np.zeros((4,4))
        T[3, 3] = 1
        T[:3, :3] = R.from_quat(quat).as_matrix()
        
        y90 = from_rotation_matrix(np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]))
        
        self.wptPokeOut[:3] = (T @ np.array([self.cuboidX, 0, 0, 0]).reshape((4,1)))[:3, 0] \
                               + obj_poses[self.cuboid][:3]
        self.wptPokeOut[3:7] = as_float_array(y90 * quaternion(quat[0], quat[1], quat[2], quat[3]))

        self.wptPokeIn[:3] = (T @ np.array([self.cuboidX/2 - poke_amount, 0, 0, 0]).reshape((4,1)))[:3, 0] \
                               + obj_poses[self.cuboid][:3]
        self.wptPokeIn[3:7] = as_float_array(y90 * quaternion(quat[0], quat[1], quat[2], quat[3]))

        self.state = State.SEARCH_WPTPOKEOUT
        
    def search_wptPokeOut(self, obs, obj_poses):
        thresh = 0.03
        # TODO: quaternion error
        if np.linalg.norm(obs.gripper_pose[:3] - self.wptPokeOut[:3]) > thresh:
        #if np.linalg.norm(obs.gripper_pose[:3] - obj_poses['waypoint2'][:3]) > thresh:
            self.goal = self.wptPokeOut.tolist()
            #self.goal = obj_poses['waypoint2'].tolist()
        else:
            self.state = State.SEARCH_WPTPOKEIN

    def search_wptPokeIn(self, obs, obj_poses):    
        thresh = 0.03
        # TODO: quaternion error
        if np.linalg.norm(obs.gripper_pose[:3] - self.wptPokeIn[:3]) > thresh:
            self.gripper = [False]
            self.goal = self.wptPokeIn.tolist()
        else:
            self.state = State.CLEAR
    
    def clear(self, obs, obj_poses):
        thresh = 0.05
        # TODO: quaternion error
        if np.linalg.norm(obs.gripper_pose[:3] - self.wptPokeOut[:3]) > thresh:
            self.goal = self.wptPokeOut.tolist()
        else:
            self.state = State.HOME

    def home(self, obs):
        thresh = 0.05
        clearance = 0.3     #distance above top of block to home
        self.grasp_amount = 0.018  #grasp more into the block when value is large
        # TODO: quaternion error
        if np.linalg.norm(obs.gripper_pose[:3] - np.array(self.home_goal[:3])) > thresh:
            # self.goal = self.home_goal
            self.goal = self.home_goal
            self.goal[0] = np.mean([pose[0] for (i, pose) in obj_poses.items()])
            self.goal[1] = np.mean([pose[1] for (i, pose) in obj_poses.items()])
            self.goal[2] = np.max([pose[2] for (i, pose) in obj_poses.items()]) + clearance 
        else:
            #regrab block positions
            pull_factor = 1.35
            gripper_offset = -0.000 #move to center of gripper, not tip
            quat = obj_poses[self.cuboid][3:7]
            T = np.zeros((4,4))
            T[3, 3] = 1
            T[:3, :3] = R.from_quat(quat).as_matrix()
            # angle = np.pi-0.00001
            angle = np.pi
            z180 = from_rotation_matrix(np.array([[np.cos(angle),-np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]))
            # z180 = self.rotation_matrix('z',angle)
            self.wptGraspOut[:3] = (T @ np.array([-self.cuboidX * 0.8, 0, gripper_offset, 0]).reshape((4,1)))[:3, 0] \
                               + obj_poses[self.cuboid][:3]
            self.wptGraspOut[3:7] = as_float_array(z180 * quaternion(quat[0], quat[1], quat[2], quat[3]))

            self.wptGraspIn[:3] = (T @ np.array([-self.cuboidX/2 + self.grasp_amount, 0, gripper_offset, 0]).reshape((4,1)))[:3, 0] \
                               + obj_poses[self.cuboid][:3]
            self.wptGraspIn[3:7] = as_float_array(z180 * quaternion(quat[0], quat[1], quat[2], quat[3]))

            self.wptPull[:3] = (T @ np.array([-self.cuboidX * pull_factor, 0, gripper_offset, 0]).reshape((4,1)))[:3, 0] \
                               + obj_poses[self.cuboid][:3]
            self.wptPull[3:7] = as_float_array(z180 * quaternion(quat[0], quat[1], quat[2], quat[3]))
            
            self.state = State.SEARCH_WPTGRASPOUT
    
    def search_wptGraspOut(self, obs, obj_poses):
        thresh = 0.05
        # TODO: quaternion error
        if np.linalg.norm(obs.gripper_pose[:3] - self.wptGraspOut[:3]) > thresh:
            self.gripper = [True]
            self.goal = self.wptGraspOut.tolist()
        else:
            self.state = State.SEARCH_WPTGRASPIN
         
    def search_wptGraspIn(self, obs, obj_poses):
        thresh = 0.05
        # TODO: quaternion error
        if np.linalg.norm(obs.gripper_pose[:3] - self.wptGraspIn[:3]) > thresh:
            self.goal = self.wptGraspIn.tolist()
        else:
            self.state = State.GRASP

    def grasp(self):
        self.gripper = [False]
        self.state = State.PULL
    
    def pull(self, obs, obj_poses):
        thresh = 0.05
        # TODO: quaternion error
        if np.linalg.norm(obs.gripper_pose[:3] - self.wptPull[:3]) > thresh:
            self.goal = self.wptPull
        else:
            #calculate placement position
            xs = [pose[0] for (i, pose) in obj_poses.items()]
            ys = [pose[1] for (i, pose) in obj_poses.items()]
            
            quat = obs.gripper_pose[3:7]
            T = np.zeros((4,4))
            T[3, 3] = 1
            T[:3, :3] = R.from_quat(quat).as_matrix()
            # angle = np.pi-0.00001
            angle = np.pi
            # z180 = from_rotation_matrix(np.array([[np.cos(angle),-np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]))
            dx,dy,_ = (T @ np.array([-self.cuboidX+self.grasp_amount, 0, 0, 0]).reshape((4,1)))[:3, 0]

            x = np.mean(xs) + dx
            y = np.mean(ys) + dy
            maxHeight = [pose[2] for (i, pose) in obj_poses.items()]
            maxHeight.sort(reverse=True)

            safety_factor = 0.005    #margin above surface when placing
            difference_factor = 0.8 #difference between height of blocks, accounting for noise

            if (maxHeight[0]-maxHeight[2])<self.cuboidZ * difference_factor:
                count = 3
                z = np.mean(maxHeight[:3]) + self.cuboidZ +safety_factor    #up a level
            elif (maxHeight[0]-maxHeight[1])<self.cuboidZ * difference_factor:
                count = 2
                z = np.mean(maxHeight[:2])+ safety_factor   #same level
            else:
                count = 1
                z = maxHeight[0]+ safety_factor #same level
            self.placePose = [x,y,z]

            self.state = State.MOVE_UP

    def move_up(self,obs,obj_poses):
        up_amount = 0.001
        thresh = 0.05
        if np.linalg.norm(obs.gripper_pose[2] - (self.placePose[2]+ up_amount)) > thresh:
            self.goal = self.wptPull
            self.goal[2] = self.placePose[2] + up_amount
        else:
            self.state = State.SEARCH_PLACEMENT

    def search_placement(self, obs, obj_poses):
        thresh = 0.05
        # TODO: quaternion error
        goal = self.placePose + self.wptGraspOut[3:7].tolist()
        if np.linalg.norm(obs.gripper_pose[:3] - goal[:3]) > thresh:
            self.goal = goal
        else:
            self.state = State.PLACE

    def place(self):
        self.gripper = [True]
        thresh = 0.05
        self.rdx = 0.1  # the dist that the gripper move out 
        rdz = 0.03
        # dx,dy,dz = (T @ np.array([-self.cuboidX+self.grasp_amount, 0, 0, 0]).reshape((4,1)))[:3, 0])
        dx,dy,dz = self.transform(obs.gripper_pose, - self.rdx,0, rdz)
        self.wptAlignout = self.placePose + self.wptGraspOut[3:7].tolist()
        self.wptAlignout[0] += dx
        self.wptAlignout[1] += dy
        self.wptAlignout[2] += dz
        self.state = State.ALIGN_OUT
    
    def rotation_matrix(self,axis,angle):
        if axis == 'x':
            R = np.array([[1,0,0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
        elif axis == 'y':
            R = np.array([[np.cos(angle),0,np.sin(angle)], [0, 1, 0], [-np.sin(angle),0,np.cos(angle)]])
        elif axis == 'z':
            R = np.array([[np.cos(angle),-np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        return from_rotation_matrix(R)

    #transform relative to global frame 
    def transform(self,reference_object,dx,dy,dz):
        quat = reference_object[3:7]
        T = np.zeros((4,4))
        T[3, 3] = 1
        T[:3, :3] = R.from_quat(quat).as_matrix()
        w_dx,w_dy,w_dz = (T @ np.array([dx,dy,dz, 0]).reshape((4,1)))[:3, 0]   #vector last element is 0. point is 1
        return w_dx,w_dy,w_dz

    def align_out(self):
        thresh = 0.03
        # rdx = -self.grasp_amount-0.2
        # dx,dy,dz = (T @ np.array([-self.cuboidX+self.grasp_amount, 0, 0, 0]).reshape((4,1)))[:3, 0])
        # dx,dy,dz = self.transform(obs.gripper_pose, rdx,0, 0)
        # goal = self.placePose + self.wptGraspOut[3:7].tolist()
        # goal[0] += dx
        # goal[1] += dy
        # goal[2] += dz
        goal = self.wptAlignout
        if np.linalg.norm(obs.gripper_pose[:3] - goal[:3]) > thresh:
            self.goal = goal
        else:
            # rdx = (self.cuboidX / 2) - (self.grasp_amount + self.rdx)
            rdx = self.rdx + 0.01
            dx,dy,dz = self.transform(obs.gripper_pose, rdx,0, 0)
            self.wptAlignout = self.placePose + self.wptGraspOut[3:7].tolist()
            self.wptAlignout[0] += dx
            self.wptAlignout[1] += dy
            self.wptAlignout[2] += dz

            self.state = State.ALIGN_IN
            
    def align_in(self):
        self.gripper = [False]
        thresh = 0.05
        goal = self.wptAlignout
        #adjust later push more
        # rdx = self.grasp_amount*2+0.2+self.cuboidX/2
        # rdz = -0.05
        # dx,dy,dz = self.transform(obs.gripper_pose, rdx,0, rdz)
        # goal = self.placePose + self.wptGraspOut[3:7].tolist()
        if np.linalg.norm(obs.gripper_pose[:3] - (goal[:3])) > thresh:
            # self.goal = self.placePose + self.wptGraspOut[3:7].tolist()
            # goal[0] += dx
            # goal[1] += dy
            # goal[2] += dz
            self.goal = goal
        else:
            self.state = State.MOVE_UP2

    def move_up2(self,obs,obj_poses):
        up_amount = 0.1
        thresh = 0.05
        if np.linalg.norm(obs.gripper_pose[2] - (self.placePose[2]+up_amount)) > thresh:
            self.goal = self.placePose + self.wptGraspOut[3:7].tolist()
            self.goal[2] = self.placePose[2] + up_amount
        else:
            self.state = State.RESET
           

    # def moving_avg(self, )
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
    print('Start~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~```')
    action_mode = ActionMode(ArmActionMode.ABS_EE_POSE_PLAN)
    env = Environment(action_mode, '', ObservationConfig(), False, frequency=5, static_positions=True)
    # task = env.get_task(StackBlocks) # available tasks: EmptyContainer, PlayJenga, PutGroceriesInCupboard, SetTheTable
    task = env.get_task(PlayJenga) # available tasks: EmptyContainer, PlayJenga, PutGroceriesInCupboard, SetTheTable
    print('Finish env init~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~``')
    obj_pose_sensor = NoisyObjectPoseSensor(env)
    descriptions, obs = task.reset()
    agent = Agent(obs, obj_pose_sensor.get_poses(),task)
    
    
    while True:
        # Getting noisy object poses
        obj_poses = obj_pose_sensor.get_poses()

        for _ in range (9):
            for key, item in obj_pose_sensor.get_poses().items():
                obj_poses[key][:3] = [sum(i) for i in zip(obj_poses[key][:3], item[:3])] 
                # print(obj_poses)
        for key, item in obj_poses.items():
            obj_poses[key][:3] = [i / 10 for i in item[:3]] 
        

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
