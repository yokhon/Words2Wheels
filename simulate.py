import json
import os
from collections import defaultdict
from pathlib import Path

import gymnasium as gym
import numpy as np

from agent import RLAgent
from env_register import register_envs
from utils import append_reward_to_env


def cal_stat(full_obs, data_hz):
    dt = 1 / data_hz
    sv_speed = np.array(full_obs['sv_speed'])
    spacing = np.array(full_obs['spacing'])
    rel_speed = np.array(full_obs['rel_speed'])

    acc = np.diff(sv_speed) / dt
    jerk = np.diff(acc) / dt

    stat = {
        'Ego Vehicle Speed': sv_speed,
        'Spacing to Leading Vehicle': spacing,
        'Relative Speed to Leading Vehicle': rel_speed,
        'Ego Vehicle Acceleration': acc,
        'Ego Vehicle Jerk': jerk,
        'Ego Vehicle Positive Acceleration': acc[acc >= 0],
        'Ego Vehicle Negative Acceleration': acc[acc < 0],
        'Ego Vehicle Positive Jerk': jerk[jerk >= 0],
        'Ego Vehicle Negative Jerk': jerk[jerk < 0],
    }

    for k, v in stat.items():
        if len(v) == 0:
            stat[k] = [0]

    # Calculate time to collision (TTC), cap values at 7 according to prior research
    ttc = spacing / rel_speed
    ttc[(ttc <= 0) | (ttc > 7)] = 7
    stat['Time to Collision'] = ttc

    stat['Time Headway'] = spacing / sv_speed

    stat.update({k: np.array(v) for k, v in full_obs.items() if k.startswith('reward_')})

    return stat

def stats_postprocess(stats, method='append'):
    processed_stats = defaultdict(list)
    for stat in stats:
        for key, value in stat.items():
            if method == 'append':
                processed_stats[key].extend(value)
            elif method == 'average':
                processed_stats[key].append(np.mean(value))
            else:
                raise ValueError('Invalid method')
    return dict(processed_stats)

def get_final_stats(stats):
    stats_to_calculate = [
        ('average', np.mean),
        ('standard deviation', np.std),
        ('min', np.min),
        ('max', np.max),
        ('25% quantile', lambda x: np.percentile(x, 25)),
        ('median', np.median),
        ('75% quantile', lambda x: np.percentile(x, 75)),
    ]
    final_stats = {}
    for key, values in stats.items():
        values = np.array(values)
        final_stats[key] = {
            name: np.round(func(values), 2) for name, func in stats_to_calculate
        }
    return final_stats

def test_agent(env, agent, output_file, verbose=False):
    data_hz = env.unwrapped.data_hz
    collision_cnt = 0
    stats = []

    episode_num = len(env.unwrapped.data)
    for _ in range(episode_num):
        full_obs = defaultdict(list)
        obs, info = env.reset()
        done = False

        while not done:
            action = agent.act(obs)
            if isinstance(action, tuple):
                action = action[0]
            obs, reward, done, _, info = env.step(action)

            full_obs['spacing'].append(obs[-3])
            full_obs['sv_speed'].append(obs[-2])
            full_obs['rel_speed'].append(obs[-1])
            full_obs['reward_full'].append(reward)

            for key, value in info.items():
                if key.startswith('reward_'):
                    full_obs[key].append(value)

        if info.get('termination_reason') == 'collision':
            collision_cnt += 1

        stats.append(cal_stat(full_obs, data_hz))

    processed_stats = stats_postprocess(stats, method='average')
    final_stats = get_final_stats(processed_stats)
    final_stats['Collision Rate'] = np.round(collision_cnt / episode_num, 4)

    with open(output_file, 'w') as f:
        json.dump(final_stats, f, indent=4)

    if verbose:
        print('final_stats:', json.dumps(final_stats))

    return final_stats

def run_simu(reward_name, config):
    reward_file = Path(config['reward_path']) / f'{reward_name}.py'
    output_env_file = f'env-{reward_name}.py'
    output_env_file = append_reward_to_env(reward_file, output_env_file=output_env_file)
    register_envs(reward_name, config)

    model_save_dir = Path(config['model_path']) / reward_name
    output_dir = Path(config['stat_path']) / reward_name
    output_dir.mkdir(parents=True, exist_ok=True)

    best_model = None
    best_reward = -float('inf')

    env_test = gym.make('CF-SafeEfficientComfort-test-v0')
    env_test.action_space.seed(config['env_params']['envs_seed'])

    model_files = sorted(model_save_dir.glob('*.pth'))

    for model_file in model_files:
        print(f"Testing model {model_file}")
        output_file = output_dir / f"{model_file.stem}.json"

        agent = RLAgent(
            config['rl_algo'], env_test, None, None, '', '', params=config['rl_params']
        )
        agent.load(str(model_file))
        stats = test_agent(env_test, agent, output_file, verbose=False)

        if stats['reward_full']['average'] > best_reward:
            best_reward = stats['reward_full']['average']
            best_model = model_file

    if best_model:
        best_output_file = output_dir / f"best-{best_model.stem}.json"
        best_stats_file = output_dir / f"{best_model.stem}.json"
        best_stats_file.rename(best_output_file)

    if Path(output_env_file).exists():
        os.remove(output_env_file)


if __name__ == '__main__':
    import argparse
    from utils import load_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--reward", required=True, help="reward-id_version-id or all")
    args = parser.parse_args()
    reward_name = args.reward

    config = load_config()

    if reward_name == 'all':
        model_path = Path(config['model_path'])
        stat_path = Path(config['stat_path'])
        stat_path.mkdir(parents=True, exist_ok=True)
        rewards = {d.name for d in model_path.iterdir() if d.is_dir()}
        tested_rewards = {d.name for d in stat_path.iterdir() if d.is_dir()}
        rewards_to_test = [
            r for r in rewards - tested_rewards if r and r[0].isdigit()
        ]
        for reward in rewards_to_test:
            run_simu(reward, config)
    else:
        run_simu(reward_name, config)
