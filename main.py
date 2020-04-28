import numpy as np
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

import time
from multiprocessing.pool import ThreadPool
import threading

from action import grasp, find_cuboid_2_grasp, knock_down, move
from utils import NoisyObjectPoseSensor, moving_avg, euler_to_quat
from Agent import Agent

#def update_reward(env, action):
#    time.sleep(2)
#    s_2, r2, done2 = env.step(action, True)
#    return s_2, r2, done2

def update_reward():
    print("thread starts")
    time.sleep(2)
    print("thread finished")


def main():
    # action_mode = ActionMode(ArmActionMode.DELTA_EE_POSE) # See rlbench/action_modes.py for other action modes
    # action_mode = ActionMode(ArmActionMode.DELTA_EE_POSE_PLAN)
    print('Start~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~```')
    action_mode = ActionMode(ArmActionMode.ABS_EE_POSE_PLAN)
    #env = Environment(action_mode, '', ObservationConfig(), False, frequency=5, static_positions=True)
    env = gym.make('play_jenga-state-v0')
    # env = gym.make('play_jenga-vision-v0')
    task = env.task
    
    # task = env.get_task(PlayJenga) # available tasks: EmptyContainer, PlayJenga, PutGroceriesInCupboard, SetTheTable
    
    print('Finish env init~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~``')
    obj_pose_sensor = NoisyObjectPoseSensor(env.env)
    # obj_pose_sensor = NoisyObjectPoseSensor(env)
    # descriptions, obs = task.reset()
    # agent = Agent(obs, obj_pose_sensor.get_poses(),task)
    
    AlgName = 'PPO'
    EnvName = 'ReachTarget'
    EnvType = 'rlbench'
    
    alg_params, learn_params = call_default_params(env, EnvType, AlgName)
    # alg = PPO(method='clip', **alg_params) # specify 'clip' or 'penalty' method for PPO
    alg_params['method'] = 'clip'
    alg = eval(AlgName+'(**alg_params)')
    training_steps = 120
    iterations = 2

    from rlbench.backend.conditions import JengaBuildTallerCondition
    obj_poses = moving_avg(obj_pose_sensor)
    task._task.register_success_conditions([JengaBuildTallerCondition( \
        env.env._scene._active_task.get_base().get_objects_in_tree(exclude_base=True, first_generation_only=False))])

    #pool = ThreadPool(processes=1)
    quat = euler_to_quat(np.pi,0,0)
    home = np.array([0.2,0.2,1]+quat.tolist()+[True])

    for it in range(iterations):
        print("BIG LOOP:",it)
        # forward
        descriptions, obs = task.reset()
        agent = Agent(obs, obj_pose_sensor.get_poses(),task)
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
            
        obs = knock_down(task)
        
        train_episodes=2
        max_steps=2 
        # save_interval=10
        save_interval=1
        gamma=0.9
        # batch_size=32
        batch_size=2
        a_update_steps=10
        c_update_steps=10

        t0 = time.time()        
        print('Training...  | Algorithm: {}  | Environment: {}'.format(alg.name, env.spec.id))
        reward_buffer = []

        for ep in range(1, train_episodes + 1):
            print("episode: ",ep)
            #s = env.reset()
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_rs_sum = 0
            visited = []
            s = env._extract_obs(obs)
            for t in range(max_steps):  # in one episode
                print("iteration: ",t)
                obj_poses = moving_avg(obj_pose_sensor)
                obs,cuboid = find_cuboid_2_grasp(obs, obj_poses, visited, task)
                obs = grasp(obs, task)


                #since gripper open/close will be performed before moving, we separate into 2 steps
                #use moving action to update reward
                #use state,reward,done after block is dropped
                a1 = alg.get_action(s)
                a2 = a1.copy()
                a2[-1] = True
                
                s_1, r1, done1, _ = env.step(a1)            #move
                env.step(a2)                                #open gripper
                env.step(home)                              #home, wait for block to drop
                s_2, r2, done2, _ = env.step(home, True)    #calc reward. when 2nd argument is true,will check success condition                

                print("REWARD",r2)

                if type(s_1) == str and s_1 == 'path':    #redo if path not found
                    print("Path Not Found!")
                    continue
                
                buffer_s.append(s)
                buffer_a.append(a1)
                buffer_r.append(r2)
                s = s_2
                ep_rs_sum += r2

                # update ppo
                if (t + 1) % batch_size == 0 or t == max_steps - 1 or done2:
                    print("updating parameters")
                    try:
                        v_s_ = alg.get_v(s_2)
                    except:
                        v_s_ = alg.get_v(s_2[np.newaxis, :])   # for raw-pixel input
                    discounted_r = []
                    for r in buffer_r[::-1]:
                        v_s_ = r + gamma * v_s_
                        discounted_r.append(v_s_)
                    discounted_r.reverse()
                    bs = buffer_s if len(buffer_s[0].shape)>1 else np.vstack(buffer_s) # no vstack for raw-pixel input
                    ba, br = np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
                    buffer_s, buffer_a, buffer_r = [], [], []
                    alg.update(bs, ba, br, a_update_steps, c_update_steps)
                if done2:
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
                plot_save_log(reward_buffer, algorithm_name=alg.name, env_name=env.spec.id)

        # reset using RL
        # alg.learn(env=env, train_episodes=3, max_steps=training_steps, save_interval=40, mode='train', render=True, **learn_params)
        # alg.learn(env=env, mode='train', **learn_params)

    alg.save_ckpt(env_name=env.spec.id)
    plot_save_log(reward_buffer, algorithm_name=alg.name, env_name=env.spec.id)

    print('Done')
    env.close()
    #env.shutdown()

    # to test:
    # alg.learn(env=env, mode='test', render=True, **learn_params)

if __name__ == "__main__":
    main()
