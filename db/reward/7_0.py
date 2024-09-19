def _get_reward(self):
    safety_weight = 0.3849  # larger weight indicates more emphasis on safety
    efficiency_weight = 0.5117  # larger weight indicates more emphasis on efficiency
    comfort_weight = 1.2712  # larger weight indicates more emphasis on comfort

    collision_penalty = -10  # introduce a large penalty for preventing collision
    TTC_threshold = 4  # larger threshold indicates larger safety zone
    jerk_const = 1  # larger jerk_const indicates larger penalty of jerk
    mu = 0.4  # mean of the log time_headway, larger mu indicates larger headway wanted
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
    rewards['safety'] += TTC_penalty

    if time_headway <= 0:
        rewards['safety'] += collision_penalty
        headway_reward = 0
    else:
        headway_reward = np.exp(-(np.log(time_headway) - mu) ** 2 / (2 * sigma ** 2)) / (
                time_headway * sigma * np.sqrt(2 * np.pi))

    rewards['safety'] *= safety_weight
    rewards['efficiency'] = headway_reward * efficiency_weight
    rewards['comfort'] = jerk_penalty * comfort_weight
    rewards['bias'] = -0.3429
    return rewards
