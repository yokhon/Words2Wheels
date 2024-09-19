import logging
import os
import pprint
from functools import partial

import numpy as np
import torch
from torch import nn
from torch.distributions import Independent, Normal
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, ReplayBuffer, VectorReplayBuffer, Batch
from tianshou.policy import PPOPolicy
from tianshou.trainer import onpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net, ActorCritic
from tianshou.utils.net.continuous import Critic, ActorProb

from geneticalgorithm import geneticalgorithm as ga


class LearningAgent:
    def __init__(
        self, model_name, train_env, valid_env, tensorboard_dir, save_path, params
    ):
        self.model_name = model_name
        self.train_env = train_env
        self.valid_env = valid_env
        self.tensorboard_dir = tensorboard_dir
        self.save_path = save_path
        self.params = params

    def train(self, tb_log_name):
        raise NotImplementedError

    def act(self, obs):
        raise NotImplementedError

    def load(self, path):
        raise NotImplementedError

    def _start_train(self, tb_log_name, validation=False):
        logging.info(f"Training model {tb_log_name}")
        os.makedirs(self.save_path, exist_ok=True)
        eval_dir = ""
        if validation:
            eval_dir = os.path.join(self.save_path, f"eval_{tb_log_name}")
            os.makedirs(eval_dir, exist_ok=True)
        save_name = os.path.join(self.save_path, f"{tb_log_name}.zip")
        return save_name, eval_dir


class RLAgent(LearningAgent):
    def __init__(
        self,
        model_name,
        env,
        train_envs,
        valid_envs,
        tensorboard_dir,
        save_path,
        params: dict,
    ):
        super().__init__(
            model_name, train_envs, valid_envs, tensorboard_dir, save_path, params
        )
        self.env = env
        self.policy = None

    def _setup_ppo_policy(self):
        state_shape = (
            self.env.observation_space.shape or self.env.observation_space.n
        )
        action_shape = (
            self.env.action_space.shape or self.env.action_space.n
        )

        # Model setup
        net_a = Net(
            state_shape,
            hidden_sizes=self.params["hidden_sizes"],
            activation=nn.Tanh,
        )
        actor = ActorProb(net_a, action_shape, unbounded=True)
        net_c = Net(
            state_shape,
            hidden_sizes=self.params["hidden_sizes"],
            activation=nn.Tanh,
        )
        critic = Critic(net_c)
        actor_critic = ActorCritic(actor, critic)

        # Initialize weights and biases
        nn.init.constant_(actor.sigma_param, -0.5)
        for m in actor_critic.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        for m in actor.mu.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)
                m.weight.data.mul_(0.01)

        optim = torch.optim.Adam(actor_critic.parameters(), lr=self.params["lr"])

        lr_scheduler = None
        if self.params.get("lr_decay", False):
            max_update_num = int(
                np.ceil(
                    self.params["step_per_epoch"] / self.params["step_per_collect"]
                )
                * self.params["max_epoch"]
            )
            lr_scheduler = LambdaLR(
                optim, lr_lambda=lambda epoch: 1 - epoch / max_update_num
            )

        def dist(*logits):
            return Independent(Normal(*logits), 1)

        self.policy = PPOPolicy(
            actor,
            critic,
            optim,
            dist,
            max_batchsize=self.params["max_batchsize"],
            discount_factor=self.params["gamma"],
            gae_lambda=self.params["gae_lambda"],
            max_grad_norm=self.params["max_grad_norm"],
            vf_coef=self.params["vf_coef"],
            ent_coef=self.params["ent_coef"],
            action_scaling=self.params["action_scaling"],
            action_bound_method=self.params["action_bound_method"],
            lr_scheduler=lr_scheduler,
            action_space=self.env.action_space,
            eps_clip=self.params["eps_clip"],
            value_clip=self.params["value_clip"],
            dual_clip=self.params["dual_clip"],
            advantage_normalization=self.params["norm_adv"],
            recompute_advantage=self.params["recompute_adv"],
        )

    def _train_ppo(self, tb_log_name):
        save_name, _ = self._start_train(tb_log_name, validation=False)

        # Set seeds for reproducibility
        np.random.seed(self.params["train_seed"])
        torch.manual_seed(self.params["train_seed"])

        self._setup_ppo_policy()

        # Collectors
        if self.params['training_env_num'] > 1:
            buffer = VectorReplayBuffer(self.params['buffer_size'], len(self.train_env))
        else:
            buffer = ReplayBuffer(self.params['buffer_size'])
        train_collector = Collector(
            self.policy, self.train_env, buffer, exploration_noise=False
        )
        test_collector = Collector(
            self.policy, self.valid_env, exploration_noise=False
        )

        # Logger setup
        log_path = os.path.join(self.tensorboard_dir, tb_log_name)
        writer = SummaryWriter(log_path)
        writer.add_text("params", str(self.params))
        logger = TensorboardLogger(writer)

        def save_best_fn(policy):
            torch.save(policy.state_dict(), save_name[:-3] + "pth")

        # Training
        result = onpolicy_trainer(
            self.policy,
            train_collector,
            test_collector,
            self.params["max_epoch"],
            self.params["step_per_epoch"],
            self.params["repeat_per_collect"],
            self.params["episode_per_test"],
            self.params["batch_size"],
            step_per_collect=self.params["step_per_collect"],
            save_best_fn=save_best_fn,
            logger=logger,
            test_in_train=False,
        )
        logging.info(pprint.pformat(result))

    def train(self, tb_log_name):
        if self.model_name == "PPO":
            self._train_ppo(tb_log_name)
        else:
            raise NotImplementedError

    def act(self, obs):
        if self.model_name == "PPO":
            obs_batch = Batch(obs=np.array([obs]), info=None)
            action = self.policy(obs_batch).act
            action = action.cpu().numpy()
            action = self.policy.map_action(action)[0]
            return action
        else:
            raise NotImplementedError

    def load(self, path):
        if self.model_name == "PPO":
            self._setup_ppo_policy()
            self.policy.load_state_dict(torch.load(path))
        else:
            raise NotImplementedError


class ConventionAgent:
    collision_penalty = 100

    def __init__(
        self,
        cf_data,
        ga_param: dict,
        ga_varbound,
        data_hz=25,
        acc_range=(-3.0, 3.0),
    ):
        self.cf_data = cf_data
        self.ga_param = ga_param
        self.ga_varbound = ga_varbound
        self.data_hz = data_hz
        self.sampling_interval = 1.0 / self.data_hz
        self._model_para = None
        self.model_name = None
        self.acc_range = acc_range

    @staticmethod
    def _act(para, spacing, svSpd, relSpd):
        raise NotImplementedError

    def act(self, obs):
        acc = np.float32(self._act(self._model_para, *obs[-3:]))
        clipped_acc = np.clip(acc, self.acc_range[0], self.acc_range[1])
        return clipped_acc / self.acc_range[1]

    def set_model_para(self, para: np.array):
        self._model_para = para

    @staticmethod
    def simulate_car_fol(
        model_fun,
        sampling_interval,
        check_collision,
        lvSpd,
        init_s,
        init_svSpd,
        para,
    ):
        """
        Simulate a car following episode based on a car-following model.
        """
        svSpd_sim = []
        spacing_sim = []
        spacing, svSpd, relSpd = init_s, init_svSpd, lvSpd[0] - init_svSpd

        svSpd_sim.append(svSpd)
        spacing_sim.append(spacing)

        for i in range(1, len(lvSpd)):
            acc = model_fun(para, spacing, svSpd, relSpd)
            svSpd_ = max(1e-3, svSpd + acc * sampling_interval)
            relSpd_ = lvSpd[i] - svSpd_
            spacing_ = spacing + sampling_interval * (relSpd_ + relSpd) / 2

            svSpd, relSpd, spacing = svSpd_, relSpd_, spacing_

            if check_collision and spacing <= 0:
                break

            svSpd_sim.append(svSpd)
            spacing_sim.append(spacing)

        return np.array(svSpd_sim), np.array(spacing_sim)

    @staticmethod
    def evaluate(simulate_func, data, penalty, para):
        error = 0
        for i in range(len(data)):
            spacing_gt = data[i][0, :]
            svSpd_gt = data[i][1, :]
            lvSpd_obs = data[i][3, :]
            init_spacing = spacing_gt[0]
            init_svSpd = svSpd_gt[0]
            svSpd_sim, spacing_sim = simulate_func(
                lvSpd_obs, init_spacing, init_svSpd, para
            )
            remain_len = len(spacing_gt) - len(spacing_sim)
            error += np.mean(
                (spacing_sim - spacing_gt[: len(spacing_sim)]) ** 2
            ) + remain_len * penalty
        return error / len(data)

    def calibrate(self, check_collision=False):
        logging.info(
            f"Calibrating {self.model_name} model with setting check_collision {check_collision}..."
        )
        simulate_partial_func = partial(
            self.simulate_car_fol,
            self._act,
            self.sampling_interval,
            check_collision,
        )
        fitness_func = partial(
            self.evaluate,
            simulate_partial_func,
            self.cf_data,
            self.collision_penalty,
        )
        model = ga(
            function=fitness_func,
            dimension=len(self.ga_varbound),
            variable_type="real",
            variable_boundaries=self.ga_varbound,
            algorithm_parameters=self.ga_param,
        )
        model.run()
        self._model_para = model.best_variable
        return model.output_dict


class IDMAgent(ConventionAgent):
    def __init__(
        self,
        cf_data,
        ga_param: dict,
        ga_varbound,
        data_hz=25,
        acc_range=(-3.0, 3.0),
        idm_params=None,
    ):
        super().__init__(cf_data, ga_param, ga_varbound, data_hz, acc_range)
        self.model_name = "IDM"
        # Load calibrated parameters
        if idm_params is not None:
            self._model_para = np.array(idm_params)
        else:
            self._model_para = None  # Should be set later

    @staticmethod
    def _act(para, spacing, svSpd, relSpd):
        """
        Function that takes IDM parameters and car-following states as inputs,
        and outputs the acceleration for the following vehicle.
        """
        (
            desiredSpd,
            desiredTimeHdw,
            maxAcc,
            comfortAcc,
            beta,
            jamSpace,
        ) = para

        desiredSpacing = jamSpace + max(
            0,
            desiredTimeHdw * svSpd
            - svSpd * relSpd / (2 * np.sqrt(maxAcc * comfortAcc)),
        )
        acc = maxAcc * (
            1 - (svSpd / desiredSpd) ** beta - (desiredSpacing / spacing) ** 2
        )

        return acc
