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

from action import grasp, find_cuboid_2_grasp, knock_down
from utils import NoisyObjectPoseSensor, moving_avg
from Agent import Agent

def main():
    # action_mode = ActionMode(ArmActionMode.DELTA_EE_POSE) # See rlbench/action_modes.py for other action modes
    # action_mode = ActionMode(ArmActionMode.DELTA_EE_POSE_PLAN)
    print('Start~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~```')
    action_mode = ActionMode(ArmActionMode.ABS_EE_POSE_PLAN)
    #env = Environment(action_mode, '', ObservationConfig(), False, frequency=5, static_positions=True)
    env = gym.make('play_jenga-state-v0')
    task = env.task
    
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
                obs = find_cuboid_2_grasp(obs, obj_poses, visited, task)
                obs = grasp(obs, task)

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

    alg.save_ckpt(env_name=env.spec.id)
    plot_save_log(reward_buffer, algorithm_name=alg.name, env_name=env.spec.id)

    print('Done')
    env.close()
    #env.shutdown()

    # to test:
    # alg.learn(env=env, mode='test', render=True, **learn_params)

if __name__ == "__main__":
    main()
