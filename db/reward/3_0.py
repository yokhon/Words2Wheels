def _get_reward(self):
    TTC_penalty_weight = 1.0  # larger weight on TTC penalty indicates more emphasis on safety
    headway_reward_weight = 1.0  # larger weight on headway reward indicates more emphasis on efficiency
    jerk_penalty_weight = 1.0  # larger weight on jerk penalty indicates more emphasis on comfort

    collision_penalty = -100  # introduce a large penalty for preventing collision
    TTC_threshold = 7  # larger threshold indicates larger safety zone
    jerk_const = 2  # larger jerk_const indicates larger penalty of jerk
    mu = 0.8  # mean of the log time_headway, larger mu indicates larger headway wanted
    sigma = 0.4  # standard deviation of the log time_headway

    rewards = {'safety': 0.0, 'efficiency': 0.0, 'comfort': 0.0}

    if self._scenario_idx == self.his_horizon:
        return rewards

    spacing, ego_speed, rel_speed = self._sim_scenario[self._scenario_idx - 1][:3]
    time_headway = spacing / ego_speed if ego_speed != 0 else float('inf')
    self.TTC = -spacing / rel_speed if rel_speed != 0 else float('inf')
    jerk_penalty = -(self._last_jerk ** 2) * jerk_const / (150 ** 2)

    TTC_penalty = (
        np.log(self.TTC / TTC_threshold)
        if 0 <= self.TTC <= TTC_threshold else 0
    )

    if time_headway <= 0:
        rewards['safety'] += collision_penalty
        headway_reward = 0
    else:
        headway_reward = np.exp(-(np.log(time_headway) - mu) ** 2 / (2 * sigma ** 2)) / (
                time_headway * sigma * np.sqrt(2 * np.pi))

    rewards['safety'] += TTC_penalty * TTC_penalty_weight
    rewards['efficiency'] = headway_reward * headway_reward_weight
    rewards['comfort'] = jerk_penalty * jerk_penalty_weight
    return rewards
