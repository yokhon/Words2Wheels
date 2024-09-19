def _get_reward(self):
    collision_penalty_weight = 0.25  # larger weight on collision penalty indicates more emphasis on safety
    headway_reward_weight = 1.0  # larger weight on headway reward indicates more emphasis on efficiency
    jerk_penalty_weight = 1.0  # larger weight on jerk penalty indicates more emphasis on comfort

    mu = 0.4  # mean of the log time_headway, larger mu indicates larger headway wanted
    sigma = 0.4  # standard deviation of the log time_headway

    rewards = {'safety': 0.0, 'efficiency': 0.0, 'comfort': 0.0}
    if self._scenario_idx == self.his_horizon:
        return rewards

    spacing, ego_speed, rel_speed = self._sim_scenario[self._scenario_idx - 1][:3]

    time_headway = spacing / ego_speed if ego_speed != 0 else float('inf')
    if time_headway <= 0:
        headway_reward = 0
    else:
        headway_reward = np.exp(-(np.log(time_headway) - mu) ** 2 / (2 * sigma ** 2)) / (
                time_headway * sigma * np.sqrt(2 * np.pi))
    rewards['efficiency'] = headway_reward * headway_reward_weight

    if spacing <= 0:
        collision_penalty = -(ego_speed ** 2)
        rewards['safety'] = collision_penalty * collision_penalty_weight

    jerk_penalty = np.exp(-np.abs(self._last_jerk)) - 1
    rewards['comfort'] = jerk_penalty * jerk_penalty_weight
    return rewards