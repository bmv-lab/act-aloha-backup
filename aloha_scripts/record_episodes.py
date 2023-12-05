import os
import time
import h5py
import argparse
import numpy as np
from tqdm import tqdm
import cProfile
import pstats
from constants import DT, HOME_POSE, TASK_CONFIGS, MASTER_IP, FOLLOWER_IP
from robot_utils import Recorder, ImageRecorder
from real_env import make_real_env
from teleop import Master, Follower, generateTrajectory
from curtsies import Input


def opening_ceremony(master,puppet):
    """ Move puppet robot to a pose where it is easy to start demonstration """

    # move arms to starting position
    puppet.move2Home()
    
    # move follower arm to same position as master
    masterJoints = master.getJointAngles()
    followerJoints = puppet.getJointAngles()
    safeTrajectory = generateTrajectory(masterJoints,followerJoints)
    for joint in safeTrajectory:
        puppet.operate(joint)

    print(f'hold c to record')
    # pressed = False
    # while not pressed:
    #     if str(input()) == 'c':
    #         pressed = True
    #     time.sleep(DT/10)
    # print(f'Started!')


def capture_one_episode(dt, max_timesteps, camera_names, dataset_dir, dataset_name, overwrite):
    print(f'Dataset name: {dataset_name}')

    master = Master(MASTER_IP)
    master.connect()

    env = make_real_env()

    # saving dataset
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    dataset_path = os.path.join(dataset_dir, dataset_name)
    if os.path.isfile(dataset_path) and not overwrite:
        print(f'Dataset already exist at \n{dataset_path}\nHint: set overwrite to True.')
        exit()

    # move puppet robot to a starting pose where it is easy to start teleoperation, then wait till both gripper closed
    opening_ceremony(master, env.puppet)

    # Data collection
    ts = env.reset(fake=True)
    timesteps = [ts]
    actions = []
    actual_dt_history = []
    
    t_start = time.time()
    # for t in tqdm(range(max_timesteps)):
    progress_bar = tqdm(total=max_timesteps)
    counter = 0
    with Input(keynames='curses') as input_generator:
        while(counter < max_timesteps):
            t0 = time.time()
            # get joints for puppet to follow
            # action = get_action(master)
            action = master.getJointAngles()
            t1 = time.time() #
            ts = env.step(action)
            t_elapsed= time.time()
            if ((t_elapsed- t_start) > 0.02) and ('c' in input_generator):
                t2 = time.time() #
                timesteps.append(ts)
                actions.append(action)
                actual_dt_history.append([t0, t1, t2])
                counter+= 1
                progress_bar.update(1)
                t_start = time.time()



    freq_mean = print_dt_diagnosis(actual_dt_history)
    # if freq_mean < 42:
    #     return False

    """
    For each timestep:
    observations
    - images
        - cam_high          (480, 640, 3) 'uint8'
        - cam_low           (480, 640, 3) 'uint8'
        - cam_wrist         (480, 640, 3) 'uint8'
    - qpos                  (14,)         'float64'
    - qvel                  (14,)         'float64'
    
    action                  (14,)         'float64'
    """

    data_dict = {
        '/observations/qpos': [],
        '/observations/qvel': [],
        '/observations/effort': [],

        '/action': [],
    }
    for cam_name in camera_names:
        data_dict[f'/observations/images/{cam_name}'] = []

    # len(action): max_timesteps, len(time_steps): max_timesteps + 1
    while actions:
        action = actions.pop(0)
        ts = timesteps.pop(0)
        data_dict['/observations/qpos'].append(ts.observation['qpos'])
        data_dict['/observations/qvel'].append(ts.observation['qvel'])
        data_dict['/observations/effort'].append(ts.observation['effort'])
        data_dict['/action'].append(action)
        for cam_name in camera_names:
            data_dict[f'/observations/images/{cam_name}'].append(ts.observation['images'][cam_name])

    # HDF5
    t0 = time.time()
    with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024**2*2) as root:
        root.attrs['sim'] = False
        obs = root.create_group('observations')
        image = obs.create_group('images')
        for cam_name in camera_names:
            _ = image.create_dataset(cam_name, (max_timesteps, 480, 640, 3), dtype='uint8',
                                     chunks=(1, 480, 640, 3), )
            # compression='gzip',compression_opts=2,)
            # compression=32001, compression_opts=(0, 0, 0, 0, 9, 1, 1), shuffle=False)
        _ = obs.create_dataset('qpos', (max_timesteps, 6))                                 # Change to 6
        _ = obs.create_dataset('qvel', (max_timesteps, 6))
        _ = obs.create_dataset('effort', (max_timesteps, 6))
        _ = root.create_dataset('action', (max_timesteps, 6))

        for name, array in data_dict.items():
            # print(name, array)
            root[name][...] = array
    print(f'Saving: {time.time() - t0:.1f} secs')

    return True


def main(args):
    task_config = TASK_CONFIGS[args['task_name']]
    dataset_dir = task_config['dataset_dir']
    max_timesteps = task_config['episode_len']
    camera_names = task_config['camera_names']

    if args['episode_idx'] is not None:
        episode_idx = args['episode_idx']
    else:
        episode_idx = get_auto_index(dataset_dir)
    overwrite = True

    dataset_name = f'episode_{episode_idx}'
    print(dataset_name + '\n')
    while True:
        is_healthy = capture_one_episode(DT, max_timesteps, camera_names, dataset_dir, dataset_name, overwrite)
        if is_healthy:
            break


def get_auto_index(dataset_dir, dataset_name_prefix = '', data_suffix = 'hdf5'):
    max_idx = 1000
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    for i in range(max_idx+1):
        if not os.path.isfile(os.path.join(dataset_dir, f'{dataset_name_prefix}episode_{i}.{data_suffix}')):
            return i
    raise Exception(f"Error getting auto index, or more than {max_idx} episodes")


def print_dt_diagnosis(actual_dt_history):
    actual_dt_history = np.array(actual_dt_history)
    get_action_time = actual_dt_history[:, 1] - actual_dt_history[:, 0]
    step_env_time = actual_dt_history[:, 2] - actual_dt_history[:, 1]
    total_time = actual_dt_history[:, 2] - actual_dt_history[:, 0]

    dt_mean = np.mean(total_time)
    dt_std = np.std(total_time)
    freq_mean = 1 / dt_mean
    print(f'Avg freq: {freq_mean:.2f} Get action: {np.mean(get_action_time):.3f} Step env: {np.mean(step_env_time):.3f}')
    return freq_mean

def debug():
    print(f'====== Debug mode ======')
    recorder = Recorder('right', is_debug=True)
    image_recorder = ImageRecorder(init_node=True, is_debug=True)
    while True:
        time.sleep(1)
        recorder.print_diagnostics()
        image_recorder.print_diagnostics()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', action='store', type=str, help='Task name.', required=True)
    parser.add_argument('--episode_idx', action='store', type=int, help='Episode index.', default=None, required=False)
    main(vars(parser.parse_args()))
    # debug()


