import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box


class CarFollowingBasicEnv(gym.Env):
    def __init__(
        self,
        cf_data,
        acc_range,
        reaction_time,
        data_hz,
        his_horizon,
        env_seed,
        reset_random=True,
        collision_avoidance=False,
    ):
        self.data = cf_data
        self.acc_range_min, self.acc_range_max = acc_range
        self.reaction_time = reaction_time

        self.data_hz = data_hz
        self.sampling_interval = 1.0 / self.data_hz
        self.reset_random = reset_random
        self.his_horizon = his_horizon
        self.collision_avoidance = collision_avoidance

        self._data_idx = 0
        self._scenario = None
        self._sim_scenario = None
        self._scenario_idx = None
        self._leading_vehicle_speeds = None
        self._last_acc = self.acc_range_min - 1.0
        self._last_jerk = 0.0

        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(3 * self.his_horizon,), dtype=np.float64
        )
        self.action_space = Box(low=-1, high=1, shape=(1,), dtype=np.float64)

        self.np_random = np.random.default_rng(env_seed)

    def _get_obs(self):
        obs = self._sim_scenario[
            self._scenario_idx - self.his_horizon : self._scenario_idx
        ].flatten()
        return obs

    def _get_reward(self):
        raise NotImplementedError

    def get_reward(self):
        reward = self._get_reward()
        if isinstance(reward, dict):
            reward = sum(reward.values())
        return reward

    def _get_info(self):
        info = {"sampling_interval": self.sampling_interval}
        reward = self._get_reward()
        if isinstance(reward, dict):
            reward_info = {f"reward_{k}": v for k, v in reward.items()}
            info.update(reward_info)
        return info

    def _action_process(self, action):
        raise NotImplementedError

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        if self.reset_random:
            idx = self.np_random.integers(0, len(self.data))
        else:
            idx = self._data_idx % len(self.data)
            self._data_idx += 1

        self._scenario = np.stack(self.data[idx]).T
        self._sim_scenario = self._scenario[:, :-1].copy()
        self._scenario_idx = self.his_horizon
        self._leading_vehicle_speeds = self._scenario[:, 3]
        self._last_acc = self.acc_range_min - 1.0

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        acc = self._action_process(action)
        spacing, sv_spd, rel_spd = self._sim_scenario[self._scenario_idx - 1]

        if self.collision_avoidance:
            safe_distance = (
                -rel_spd * self.reaction_time
                + rel_spd**2 / (2 * self.acc_range_max)
            )
            if spacing < safe_distance:
                acc = self.acc_range_min

        last_acc = acc if self._last_acc < self.acc_range_min else self._last_acc
        self._last_jerk = (acc - last_acc) / self.sampling_interval
        self._last_acc = acc

        sv_spd_ = max(1e-4, sv_spd + acc * self.sampling_interval)
        rel_spd_ = self._leading_vehicle_speeds[self._scenario_idx] - sv_spd_
        spacing_ = spacing + self.sampling_interval * (rel_spd_ + rel_spd) / 2

        self._sim_scenario[self._scenario_idx] = np.array([spacing_, sv_spd_, rel_spd_])
        self._scenario_idx += 1

        reward = self.get_reward()
        observation = self._get_obs()
        info = self._get_info()

        if spacing_ <= 0:
            terminated = True
            info["termination_reason"] = "collision"
        elif self._scenario_idx >= len(self._scenario):
            terminated = True
            info["termination_reason"] = "end of scenario"
        else:
            terminated = False
            info["termination_reason"] = "none"
        return observation, reward, terminated, False, info


class CarFollowingSafeEfficientComfortEnv(CarFollowingBasicEnv):
    def __init__(
        self,
        cf_data,
        acc_range,
        reaction_time,
        data_hz,
        his_horizon,
        env_seed,
        reset_random=True,
        collision_avoidance=False,
    ):
        super().__init__(
            cf_data,
            acc_range,
            reaction_time,
            data_hz,
            his_horizon,
            env_seed,
            reset_random,
            collision_avoidance,
        )

    def _action_process(self, action):
        action = np.asarray(action).squeeze()
        return action * self.acc_range_max
