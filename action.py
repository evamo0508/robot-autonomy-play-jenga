import numpy as np
from scipy.spatial.transform import Rotation as R
from quaternion import from_rotation_matrix, quaternion, as_float_array
from pyrep.objects.shape import Shape

from utils import euler_to_quat 

def move(x,y,z,roll,pitch,yaw,task):
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
    
def grasp(obs, task):
    action = obs.gripper_pose[:7].tolist() + [False]
    obs,_,_ = task.step(action)
    action[2] += 0.3
    obs,_,_ = task.step(action)
    return obs

def find_cuboid_2_grasp(obs, obj_poses, visited, task):
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
    obs1 = move(wptPick[0],wptPick[1],wptPick[2],orientation[0],orientation[1],orientation[2], task)
    # obs1,_,_ = task.step(action)
    if obs1 == []:
        if len(visited) == 15:
            print('lalalalalalalalal')
            env.shutdown()
        return find_cuboid_2_grasp(obs,obj_poses, visited, task)
    return obs1

