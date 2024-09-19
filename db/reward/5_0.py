def _get_reward(self):
    """Human-like driving reward that minimizes the difference between generated behavior and natural behavior."""

    spacing_diff_weight = 0.0
    speed_diff_weight = 1.0

    rewards = {'spacing_diff': 0.0, 'speed_diff': 0.0,}

    sim_spacing, sim_speed = self._sim_scenario[self._scenario_idx - 1][:2]
    gt_spacing, gt_speed = self._scenario[self._scenario_idx - 1][:2]

    spacing_diff = -np.log(np.abs(sim_spacing - gt_spacing) / max(gt_spacing, 1e-3))
    speed_diff = -np.log(np.abs(sim_speed - gt_speed) / max(gt_speed, 1e-3))

    rewards['spacing_diff'] = spacing_diff * spacing_diff_weight
    rewards['speed_diff'] = speed_diff * speed_diff_weight

    return rewards
