import numpy as np
import scipy as sp
from scipy.spatial.transform import Rotation as R
from quaternion import from_rotation_matrix, quaternion, as_float_array

from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import *
from pyrep.objects.shape import Shape

import gym
import rlbench.gym
from rlzoo.common.env_wrappers import *
from rlzoo.common.utils import *
from rlzoo.algorithms import *

from enum import Enum 
import time

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
        self._jenga = task
        self.poke_out = 0.09 # the more, the farther away from block before poking
        self.place_amount = 0.06
        self.grasp_amount = 0.018  #grasp more into the block when value is large
        self.align_out_amount = -0.05  # the dist that the gripper move out before aligning. The more negative, the more it moves out
        self.align_in_amount = 0.08     #increases this pushes in more
        self.up_amount = 0.001
        self.up_amount2 = 0.42
        self.knock = False          #whether in knock down mode


        #make it die fast
        self.poke_out = 10
        # self.move_up2 = 10.0
        
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
        xmin,xmax = 0, 0.45
        ymin,ymax = -0.45, 0.4
        zmin,zmax = 0.77,1.7
        # print (self.goal[:3])
        if self.goal[0] < xmin or self.goal[0] > xmax or self.goal[1] < ymin or self.goal[1] > ymax or self.goal[2] < zmin or self.goal[2]> zmax:
            print("OUT OF BOUNDS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!") 
            self.knock = True 
        return self.goal + self.gripper, self.knock

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
        self.knock = False
         
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
        # self.cuboid = 'Cuboid5'
        # self.cuboid = 'Cuboid3'
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
        
        self.wptPokeOut[:3] = (T @ np.array([self.cuboidX/2+self.poke_out, 0, 0, 0]).reshape((4,1)))[:3, 0] \
                               + obj_poses[self.cuboid][:3]
        self.wptPokeOut[3:7] = as_float_array(y90 * quaternion(quat[0], quat[1], quat[2], quat[3]))

        self.wptPokeIn[:3] = (T @ np.array([self.cuboidX/2 - poke_amount, 0, 0, 0]).reshape((4,1)))[:3, 0] \
                               + obj_poses[self.cuboid][:3]
        self.wptPokeIn[3:7] = as_float_array(y90 * quaternion(quat[0], quat[1], quat[2], quat[3]))

        if self.wptPokeOut[2] < 0.77:
            self.knock = True 

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
            dx,dy,_ = (T @ np.array([-self.cuboidX+self.place_amount, 0, 0, 0]).reshape((4,1)))[:3, 0]

            x = np.mean(xs) + dx
            y = np.mean(ys) + dy
            maxHeight = [pose[2] for (i, pose) in obj_poses.items()]
            maxHeight.sort(reverse=True)

            safety_factor = 0.005    #margin above surface when placing
            difference_factor = 0.8 #difference between height of blocks, accounting for noise
            
            #determine height of placing position
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
        thresh = 0.05
        if np.linalg.norm(obs.gripper_pose[2] - (self.placePose[2]+ self.up_amount)) > thresh:
            self.goal = self.wptPull
            self.goal[2] = self.placePose[2] + self.up_amount
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
        
        rdz = 0.03
        # dx,dy,dz = (T @ np.array([-self.cuboidX+self.grasp_amount, 0, 0, 0]).reshape((4,1)))[:3, 0])
        dx,dy,dz = self.transform(obs.gripper_pose, + self.align_out_amount,0, rdz)
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
            # rdx = (self.cuboidX / 2) - (self.grasp_amount + self.align_out_amount)
            rdx = self.align_out_amount + self.align_in_amount
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
        thresh = 0.05
        goal = self.placePose + self.wptGraspOut[3:7].tolist()
        goal[2] = self.placePose[2] + self.up_amount2
        if np.linalg.norm(obs.gripper_pose[:3] - goal[:3]) > thresh:
            self.goal = goal
        else:
            self.state = State.RESET

def euler_to_quat(roll, pitch, yaw):
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    return np.array([qx, qy, qz, qw])

def move(x,y,z,roll,pitch,yaw):
    action = [x,y,z,0,0,0,0,True]
    action[3:7] = euler_to_quat(roll, pitch, yaw)
    obs,_,_ = task.step(action)
    return obs
    


def knock_down():
    #knock down
    roll,pitch,yaw = np.pi,0,np.pi
    obs = move(0.3,-0.4,0.9,roll,pitch,yaw)
    obs = move(0.1,0.3,0.75,roll,pitch,yaw)
    obs = move(0.2,-0.2,0.75,roll,pitch,yaw)
    
    #clear
    roll,pitch,yaw = np.pi,0,np.pi*1/2
    obs = move(0.2,-0.3,0.75,roll,pitch,yaw)
    obs = move(0.2,0.1,0.75,roll,pitch,yaw)  
    obs = move(0.1,0.1,0.75,roll,pitch,yaw)
    obs = move(0.1,-0.1,0.75,roll,pitch,yaw)
    obs = move(0.1,-0.1,1.0,roll,pitch,yaw)
    return obs
    

class NoisyObjectPoseSensor:

    def __init__(self, env):
        self._env = env

        self._pos_scale = [0.005] * 3
        self._rot_scale = [0.01] * 3
        #no noise
        self._pos_scale = [0] *3
        self._rot_scale = [0] * 3

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

def find_cuboid_2_grasp(obs, obj_poses, visited):
    cuboidX = 0.2067
    while True:
        cuboidID = np.random.choice(range(0,15))    
        if cuboidID not in visited:
            visited.append(cuboidID)
            break
    if cuboidID == 13:
        cuboid = 'target_cuboid'
    elif cuboidID == 14:
        cuboid = 'Cuboid'
    else:
        cuboid = 'Cuboid' + str(cuboidID)
    print(cuboid)
    task._task.register_graspable_objects([Shape(cuboid)])

    quat = obj_poses[cuboid][3:7]
    wptPick = [0,0,0,0,0,0,0]
    wptPick[:3] = obj_poses[cuboid][:3]
    
    
    orientation = R.from_quat(quat).as_euler('xyz', degrees = False)
    # print("orientation\n",orientation*180/np.pi)
    
    orientation[0],orientation[1] = np.pi,0
    new_quat = R.from_euler('xyz', [orientation], degrees = False).as_quat().squeeze()

    wptPick[3:] = as_float_array(quaternion(new_quat[0], new_quat[1], new_quat[2], new_quat[3]))
    action =  wptPick + [True]    
    
    move(wptPick[0],wptPick[1],wptPick[2]+0.1,orientation[0],orientation[1],orientation[2])
    obs1 = move(wptPick[0],wptPick[1],wptPick[2],orientation[0],orientation[1],orientation[2])
    # obs1,_,_ = task.step(action)
    if obs1 == []:
        if len(visited) == 15:
            print('lalalalalalalalal')
            env.shutdown()
        return find_cuboid_2_grasp(obs,obj_poses, visited)
    return obs1


def grasp(obs):
    action = obs.gripper_pose[:7].tolist() + [False]
    obs,_,_ = task.step(action)
    action[2] += 0.3
    obs,_,_ = task.step(action)
    return obs

def moving_avg(obj_pose_sensor):
    mov_avg = 50
    obj_poses = obj_pose_sensor.get_poses()
    for _ in range (mov_avg-1):
        for key, item in obj_pose_sensor.get_poses().items():
            obj_poses[key][:3] = [sum(i) for i in zip(obj_poses[key][:3], item[:3])] 
            # print(obj_poses)
    for key, item in obj_poses.items():
        obj_poses[key][:3] = [i / mov_avg for i in item[:3]] 
    return obj_poses


if __name__ == "__main__":
    # action_mode = ActionMode(ArmActionMode.DELTA_EE_POSE) # See rlbench/action_modes.py for other action modes
    # action_mode = ActionMode(ArmActionMode.DELTA_EE_POSE_PLAN)
    print('Start~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~```')
    action_mode = ActionMode(ArmActionMode.ABS_EE_POSE_PLAN)
    #env = Environment(action_mode, '', ObservationConfig(), False, frequency=5, static_positions=True)
    env = gym.make('play_jenga-state-v0')
    task = env.task
    # print(type(env))
    # print(type(task))
    # print(type(task._task))
    
    # task = env.get_task(PlayJenga) # available tasks: EmptyContainer, PlayJenga, PutGroceriesInCupboard, SetTheTable
    
    print('Finish env init~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~``')
    obj_pose_sensor = NoisyObjectPoseSensor(env.env)
    # obj_pose_sensor = NoisyObjectPoseSensor(env)
    descriptions, obs = task.reset()
    agent = Agent(obs, obj_pose_sensor.get_poses(),task)
    
    AlgName = 'PPO'
    EnvName = 'ReachTarget'
    EnvType = 'rlbench'
    
    alg_params, learn_params = call_default_params(env, EnvType, AlgName)
    # alg = PPO(method='clip', **alg_params) # specify 'clip' or 'penalty' method for PPO
    alg_params['method'] = 'clip'
    alg = eval(AlgName+'(**alg_params)')
    training_steps = 120
    iterations = 10

    from rlbench.backend.conditions import JengaBuildTallerCondition
    obj_poses = moving_avg(obj_pose_sensor)
    task._task.register_success_conditions([JengaBuildTallerCondition(obj_poses)])


    for it in range(iterations):
        # forward
        
        while True:
            # Getting noisy object poses
            obj_poses = moving_avg(obj_pose_sensor)

            # Getting various fields from obs
            current_joints = obs.joint_positions
            gripper_pose = obs.gripper_pose
            rgb = obs.wrist_rgb
            depth = obs.wrist_depth
            mask = obs.wrist_mask

            # Perform action and step simulation
            # action = agent.act(obs)
            
            action,knock = agent.act(obs, obj_poses)

            # break out action xyz is out of bounds
            
            
            # break if path not found
            obs, reward, terminate = task.step(action)
            if knock == True:
                break
            if obs == [] and reward == [] and terminate == []:
                break
            
        obs = knock_down()
        
        train_episodes=1
        max_steps=200 
        save_interval=10
        gamma=0.9
        batch_size=32
        a_update_steps=10
        c_update_steps=10
        EPS = 1e-8

        t0 = time.time()        
        print('Training...  | Algorithm: {}  | Environment: {}'.format(alg.name, env.spec.id))
        reward_buffer = []

        for ep in range(1, train_episodes + 1):
            #s = env.reset()
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_rs_sum = 0
            visited = []
            s = env._extract_obs(obs)
            for t in range(max_steps):  # in one episode
                obj_poses = moving_avg(obj_pose_sensor)
                obs = find_cuboid_2_grasp(obs, obj_poses, visited)
                obs = grasp(obs)

                a = alg.get_action(s)
                s_, r, done, _ = env.step(a)
                print("reward",r)
                if s_ == 'path':    #redo if path not found
                    continue
                
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r)
                s = s_
                ep_rs_sum += r

                # update ppo
                if (t + 1) % batch_size == 0 or t == max_steps - 1 or done:
                    try:
                        v_s_ = alg.get_v(s_)
                    except:
                        v_s_ = alg.get_v(s_[np.newaxis, :])   # for raw-pixel input
                    discounted_r = []
                    for r in buffer_r[::-1]:
                        v_s_ = r + gamma * v_s_
                        discounted_r.append(v_s_)
                    discounted_r.reverse()
                    bs = buffer_s if len(buffer_s[0].shape)>1 else np.vstack(buffer_s) # no vstack for raw-pixel input
                    ba, br = np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
                    buffer_s, buffer_a, buffer_r = [], [], []
                    alg.update(bs, ba, br, a_update_steps, c_update_steps)
                if done:
                    print("episode done!")
                    break

            print(
                'Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                    ep, train_episodes, ep_rs_sum,
                    time.time() - t0
                )
            )

            reward_buffer.append(ep_rs_sum)
            if ep and not ep % save_interval:
                alg.save_ckpt(env_name=env.spec.id)
                plot_save_log(reward_buffer, algorithm_name=alg.name, env_name=alg.spec.id)

    

        # reset using RL
        # alg.learn(env=env, train_episodes=3, max_steps=training_steps, save_interval=40, mode='train', render=True, **learn_params)
        # alg.learn(env=env, mode='train', **learn_params)

        
        """
        for i in range(training_steps):
            if i % episode_length == 0:
                print('Reset Episode')
                obs = env.reset()
            # need to assign action with the learned policy here
            # action = 
            obs, reward, terminate, _ = env.step(env.action_space.sample())
            #env.render()
            
            if terminate:
                break
        """
    alg.save_ckpt(env_name=env.spec.id)
    plot_save_log(reward_buffer, algorithm_name=alg.name, env_name=env.spec.id)

    print('Done')
    env.close()
    #env.shutdown()

    # to test:
    # alg.learn(env=env, mode='test', render=True, **learn_params)
