'''
File name: test_bc.py
Desription: This is a testing/evaluation file for ACT which works in the following manner

1) Parse arguments
2) Load model .pth file
3) Load the policy and the optimizer
4) Collect real-time feed from the robot containing qpos, actions, and images
5) Set temporal agg query frequency
6) Start evaluation

'''

import sys
directory_to_add = "~/act_dec2023"
sys.path.append(directory_to_add)

import torch
import numpy as np
import os
import pickle
import argparse
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange

from aloha_scripts.constants import DT
from aloha_scripts.constants import PUPPET_GRIPPER_JOINT_OPEN
from aloha_scripts.teleop import generateTrajectory
# from utils import load_data # data functions
from utils import sample_box_pose, sample_insertion_pose # robot functions
from utils import set_seed # helper functions
from policy import ACTPolicy
from visualize_episodes import save_videos
import time
import signal
from aloha_scripts.real_env import make_real_env


# from sim_env import BOX_POSE                          

import IPython
e = IPython.embed

def main(args):
    set_seed(1)
    # command line parameters
    is_eval = args['eval']
    ckpt_dir = args['ckpt_dir']
    policy_class = args['policy_class']
    onscreen_render = args['onscreen_render']
    task_name = args['task_name']
    batch_size_train = args['batch_size']
    batch_size_val = args['batch_size']
    num_epochs = args['num_epochs']

    # get task parameters
    is_sim = task_name[:4] == 'sim_'
    if is_sim:
        print("Inside the simulation environment param cond")              # Commmented original code block and added pass instead in simulation block
        pass
        # from constants import SIM_TASK_CONFIGS
        # task_config = SIM_TASK_CONFIGS[task_name]
    else:
        from aloha_scripts.constants import TASK_CONFIGS
        task_config = TASK_CONFIGS[task_name]
    dataset_dir = task_config['dataset_dir']
    num_episodes = task_config['num_episodes']
    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']

    # fixed parameters
    # state_dim = 14
    # lr_backbone = 1e-5
    # backbone = 'resnet18'
    if policy_class == 'ACT':
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {'lr': args['lr'],
                            'num_queries': args['chunk_size'],
                            'kl_weight': args['kl_weight'],
                            'hidden_dim': args['hidden_dim'],
                            'dim_feedforward': args['dim_feedforward'],
                            'lr_backbone': args['lr_backbone'],
                            'state_dim': args['state_dim'],
                            'backbone': args['backbone'],
                            'enc_layers': enc_layers,
                            'dec_layers': dec_layers,
                            'nheads': nheads,
                            'camera_names': camera_names,
                            }
    # elif policy_class == 'CNNMLP':                                                                                        # Commented CNNMLP policy block
    #     policy_config = {'lr': args['lr'], 'lr_backbone': lr_backbone, 'backbone' : backbone, 'num_queries': 1,
    #                      'camera_names': camera_names,}
    else:
        raise NotImplementedError

    config = {
        'num_epochs': num_epochs,
        'ckpt_dir': ckpt_dir,
        'episode_len': episode_len,
        'state_dim': args['state_dim'],
        'lr': args['lr'],
        'policy_class': policy_class,
        'onscreen_render': onscreen_render,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': args['seed'],
        'temporal_agg': args['temporal_agg'],
        'camera_names': camera_names,
        'real_robot': not is_sim                            # args on real robot
    }

    if is_eval:
        ckpt_name = f'policy_best.ckpt'
        results = []
        success_rate, avg_return = eval_bc(config, ckpt_name, save_episode=True)
        results.append([ckpt_name, success_rate, avg_return])


        for ckpt_name, success_rate, avg_return in results:
            print(f'{ckpt_name}: {success_rate=} {avg_return=}')
        print()
        exit()

            
def get_image(ts, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image        
    
def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    # elif policy_class == 'CNNMLP':                            # Commented CNNMLP policy class block
    #     policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    # elif policy_class == 'CNNMLP':                            # Commented CNNMLP policy optimizer block
    #     optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


def signal_handler(signal, frame):
    print("Ctrl-C detected. Exiting...")
    sys.exit(0)


def eval_bc(config, ckpt_name, save_episode=False):

    set_seed(1000)
    ckpt_dir = config['ckpt_dir']
    state_dim = config['state_dim']
    real_robot = config['real_robot']
    policy_class = config['policy_class']
    onscreen_render = config['onscreen_render']
    policy_config = config['policy_config']
    camera_names = config['camera_names']
    max_timesteps = config['episode_len']
    task_name = config['task_name']
    temporal_agg = config['temporal_agg']
    onscreen_cam = 'angle'

    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path))
    print(loading_status)
    policy.cuda()
    policy.eval()
    print(f'Loaded: {ckpt_path}')
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    env = make_real_env()
    env.puppet.move2Home()
    env_max_reward = 0

    query_frequency = policy_config['num_queries']
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config['num_queries']

    max_timesteps = int(max_timesteps * 1) # may increase for real-world tasks

    num_rollouts = 1
    episode_returns = []
    highest_rewards = []

    ts = env.reset()

    ### evaluation loop
    if temporal_agg:
        all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, state_dim]).cuda()

    qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
    image_list = [] # for visualization
    qpos_list = []
    target_qpos_list = []
    rewards = []
    loop_times= []
    calculation_times=[]
    safeFlag = True
    with torch.inference_mode():
        # print('query freq:     ',query_frequency)
        # time.sleep(0.5)
        for t in range(max_timesteps):
            signal.signal(signal.SIGINT, signal_handler)

            ### process previous timestep to get qpos and image_list
            tic= time.perf_counter()
            obs = ts.observation
            # x,y,z,raw,pitch,yaw = env.puppet.getTCPPosition()
            # if (z) < 0.06:
            #     env.puppet.stop()
            if 'images' in obs:
                image_list.append(obs['images'])
            else:
                image_list.append({'main': obs['image']})
            qpos_numpy = np.array(obs['qpos'])
            qpos = pre_process(qpos_numpy)
            qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
            qpos_history[:, t] = qpos
            curr_image = get_image(ts, camera_names)

            tic2= time.perf_counter()
            ### query policy
            if config['policy_class'] == "ACT":
                if t % query_frequency == 0:
                    all_actions = policy(qpos, curr_image)
                    print(all_actions.device)
                    # tic_policy_est= time.perf_counter()
                if temporal_agg:
                    all_time_actions[[t], t:t+num_queries] = all_actions
                    actions_for_curr_step = all_time_actions[:, t]
                    actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                    actions_for_curr_step = actions_for_curr_step[actions_populated]
                    k = 0.01
                    exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                    exp_weights = exp_weights / exp_weights.sum()
                    exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                    raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    # tic_temp_est= time.perf_counter()
                else:
                    raw_action = all_actions[:, t % query_frequency]
            else:
                raise NotImplementedError
            ### post-process actions
            raw_action = raw_action.squeeze(0).cpu().numpy()
            action = post_process(raw_action)
            target_qpos = action
            print("T:  ", t, "\t", "Target Pose:  ", target_qpos)
            toc= time.perf_counter()
            loop_time= toc-tic
            calculation_time= toc-tic2
            calculation_time= loop_time-(tic2-tic)
            # policy_est_time = tic_policy_est - tic
            # temp_est = tic_temp_est - tic_policy_est
            print("Loop Time:   ",loop_time , "Calculation time:   ", calculation_time)
            # print("Policy estimation time:   ",policy_est_time)
            # print("Temporal calculation time:   ",temp_est)
            loop_times.append(loop_time)
            calculation_times.append(calculation_time)
            # target_qpos[-1] = target_qpos[-1]-(np.pi/2)
            ts = env.step(target_qpos)
            time.sleep(0.02)

            ### for visualization
            qpos_list.append(qpos_numpy)
            target_qpos_list.append(target_qpos)
            rewards.append(ts.reward)
    print("############# Average loop time:    ",  np.mean(loop_times))
    print("############# Average loop time:    ",  np.mean(calculation_times))
    env.puppet.stop()

    rewards = np.array(rewards)
    episode_return = np.sum(rewards[rewards!=None])
    episode_returns.append(episode_return)
    episode_highest_reward = np.max(rewards)
    highest_rewards.append(episode_highest_reward)
    print(f'Rollout {0}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward==env_max_reward}')

    if save_episode:
        save_videos(image_list, DT, video_path=os.path.join(ckpt_dir, f'video{0}.mp4'))

    success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
    avg_return = np.mean(episode_returns)
    summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
    for r in range(env_max_reward+1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / num_rollouts
        summary_str += f'Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n'

    print(summary_str)

    # save success rate to txt
    result_file_name = 'result_' + ckpt_name.split('.')[0] + '.txt'
    with open(os.path.join(ckpt_dir, result_file_name), 'w') as f:
        f.write(summary_str)
        f.write(repr(episode_returns))
        f.write('\n\n')
        f.write(repr(highest_rewards))

    return success_rate, avg_return

        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)
    parser.add_argument('--lr_backbone', action='store', type=float, help='lr_backbone', required=True) #MAA 092723
    parser.add_argument('--state_dim', action='store', type=int, help='state_dim', required=True) #MAA 092723
    parser.add_argument('--backbone', action='store', type=str, help='backbone', required=True) #MAA 092723

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    parser.add_argument('--temporal_agg', action='store_true')
    
    main(vars(parser.parse_args()))
