import os
import datetime
from pathlib import Path

import gymnasium as gym
from tianshou.env import DummyVectorEnv

from agent import RLAgent
from utils import append_reward_to_env
from env_register import register_envs


def run_train(reward_name, n_run, config):
    reward_file = Path(config['reward_path']) / f"{reward_name}.py"
    output_env_file = f"env-{reward_name}.py"
    output_env_file = append_reward_to_env(reward_file, output_env_file=output_env_file)
    register_envs(reward_name, config)

    algo = config['rl_algo']
    for train_seed in range(1, n_run + 1):
        rl_params = config['rl_params'].copy()
        rl_params['train_seed'] = train_seed
        timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M')
        tensorboard_dir = Path(config['tensorboard_path']) / algo / reward_name / timestamp
        model_save_dir = Path(config['model_path']) / reward_name

        env = gym.make('CF-SafeEfficientComfort-train-v0')
        train_envs = DummyVectorEnv(
            [lambda: gym.make('CF-SafeEfficientComfort-train-v0') for _ in range(rl_params['training_env_num'])]
        )
        valid_envs = DummyVectorEnv(
            [lambda: gym.make('CF-SafeEfficientComfort-valid-v0') for _ in range(rl_params['training_env_num'])]
        )
        train_envs.seed(config['env_params']['envs_seed'])
        valid_envs.seed(config['env_params']['envs_seed'])

        agent = RLAgent(
            algo, env, train_envs, valid_envs, tensorboard_dir, model_save_dir, rl_params
        )
        try:
            agent.train(tb_log_name=f"seed_{train_seed}")
        except Exception as e:
            print(f"Training failed with reward {reward_name} seed {train_seed}: {e}")
            break

    if Path(output_env_file).exists():
        os.remove(output_env_file)


if __name__ == '__main__':
    import argparse
    from utils import load_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--reward", required=True, help="reward-id_version-id or all")
    parser.add_argument("--n_run", type=int, default=5, help="number of repeated trainings")
    args = parser.parse_args()
    reward_name = args.reward
    n_run = args.n_run

    config = load_config()

    if reward_name == 'all':
        reward_path = Path(config['reward_path'])
        model_path = Path(config['model_path'])
        model_path.mkdir(parents=True, exist_ok=True)
        rewards = {f.stem for f in reward_path.glob('*.py')}
        trained_rewards = {f.name for f in model_path.iterdir()}
        rewards_to_train = [r for r in rewards - trained_rewards if r and r[0].isdigit()]
        for reward in rewards_to_train:
            run_train(reward, n_run, config)
    else:
        run_train(reward_name, n_run, config)
