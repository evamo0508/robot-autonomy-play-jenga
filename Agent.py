import numpy as np
from utils import State, transform, rotation_matrix
from scipy.spatial.transform import Rotation as R
from quaternion import from_rotation_matrix, quaternion, as_float_array

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

        xmin,xmax = 0, 0.45
        ymin,ymax = -0.45, 0.4
        zmin,zmax = 0.77,1.7
        # print (self.goal[:3])
        if self.goal[0] < xmin or self.goal[0] > xmax or self.goal[1] < ymin or self.goal[1] > ymax or self.goal[2] < zmin or self.goal[2]> zmax:
            print("OUT OF BOUNDS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!") 
            self.knock = True 
        return self.goal + self.gripper, self.knock

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
            # z180 = rotation_matrix('z',angle)
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
        dx,dy,dz = transform(obs.gripper_pose, + self.align_out_amount,0, rdz)
        self.wptAlignout = self.placePose + self.wptGraspOut[3:7].tolist()
        self.wptAlignout[0] += dx
        self.wptAlignout[1] += dy
        self.wptAlignout[2] += dz
        self.state = State.ALIGN_OUT
   
    def align_out(self):
        thresh = 0.03
        # rdx = -self.grasp_amount-0.2
        # dx,dy,dz = (T @ np.array([-self.cuboidX+self.grasp_amount, 0, 0, 0]).reshape((4,1)))[:3, 0])
        # dx,dy,dz = transform(obs.gripper_pose, rdx,0, 0)
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
            dx,dy,dz = transform(obs.gripper_pose, rdx,0, 0)
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
        # dx,dy,dz = transform(obs.gripper_pose, rdx,0, rdz)
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
