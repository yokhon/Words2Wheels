import numpy as np
from gymnasium.envs.registration import register


def split_train(data, test_ratio, val_ratio, seed=42):
    np.random.seed(seed)
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    val_number = int(len(data) * (test_ratio + val_ratio))
    test_indices = shuffled_indices[:test_set_size]
    val_indices = shuffled_indices[test_set_size:val_number]
    train_indices = shuffled_indices[val_number:]
    return data[train_indices], data[test_indices], data[val_indices]

def load_data(data_path):
    return np.load(data_path, allow_pickle=True)

def register_envs(reward_name, config):
    data = load_data(config['data_path'])
    train, test, valid = split_train(
        data,
        test_ratio=0.15,
        val_ratio=0.15,
        seed=43,
    )

    env_config = config['env_params']
    env_common_kwargs = {
        'acc_range': env_config.get('acc_range', [-3.0, 3.0]),
        'reaction_time': env_config.get('reaction_time', 1.0),
        'data_hz': env_config.get('data_hz', 25),
        'his_horizon': env_config.get('his_horizon', 25),
        'env_seed': env_config.get('envs_seed', 42),
    }

    register(
        id='CF-SafeEfficientComfort-train-v0',
        entry_point=f'env-{reward_name}:CarFollowingSafeEfficientComfortEnv',
        kwargs={
            **env_common_kwargs,
            'cf_data': train,
            'reset_random': True,
        }
    )

    register(
        id='CF-SafeEfficientComfort-valid-v0',
        entry_point=f'env-{reward_name}:CarFollowingSafeEfficientComfortEnv',
        kwargs={
            **env_common_kwargs,
            'cf_data': valid,
            'reset_random': True,
        }
    )

    register(
        id='CF-SafeEfficientComfort-test-v0',
        entry_point=f'env-{reward_name}:CarFollowingSafeEfficientComfortEnv',
        kwargs={
            **env_common_kwargs,
            'cf_data': test,
            'reset_random': False,
            'collision_avoidance': True,
        }
    )
