import argparse
import glob
import importlib
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gym
import stable_baselines3 as sb3  # noqa: F401
import torch as th  # noqa: F401
import yaml
from sb3_contrib import QRDQN, TQC
from stable_baselines3 import A2C, DDPG, DQN, HER, PPO, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike  # noqa: F401
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecFrameStack, VecNormalize

# For custom activation fn
from torch import nn as nn  # noqa: F401 pylint: disable=unused-import
import torch.nn.functional as F
import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
ALGOS = {
    "a2c": A2C,
    "ddpg": DDPG,
    "dqn": DQN,
    "ppo": PPO,
    "her": HER,
    "sac": SAC,
    "td3": TD3,
    # SB3 Contrib,
    "qrdqn": QRDQN,
    "tqc": TQC,
}

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512, attn_type: str = 'attn',adaptive: bool = False):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        if attn_type == 'Attn':
            layer = MultiHeadAttn(size=32, adaptive=adaptive)
        elif attn_type == 'RvuAttn':
            layer = RvuAttn(size=32, adaptive=adaptive)
        elif attn_type == 'CrossAttn':
            layer = CrossAttn(size=32, adaptive=adaptive)
        elif attn_type == 'xAttn':
            layer = xAttn(size=32, adaptive=adaptive)
        elif attn_type == 'noAttn':
            layer = nn.Dropout(0)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            layer,
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

class MultiHeadAttn(nn.Module):
    def __init__(self, size, adaptive):
        super().__init__()
        #Approximator function for weight q,k,v
        self.w_q = nn.Conv2d(in_channels=size ,out_channels=size , kernel_size=1)
        self.w_k = nn.Conv2d(in_channels=size ,out_channels=size , kernel_size=1)
        self.w_v = nn.Conv2d(in_channels=size ,out_channels=size , kernel_size=1)
        #self.out = nn.Conv2d(in_channels=size ,out_channels=size , kernel_size=1)
        self.adaptive = adaptive
        if adaptive:
            self.W = th.nn.Parameter(th.randn(1))
            self.W.requires_grad = True

    def forward(self, a, dropout=None):
        q = self.w_q(a)
        k = self.w_k(a)
        v = self.w_v(a)

        attention = self.dot_pdt_attention(q, k, v, dropout)
        #out = F.relu(self.out(attention))
        if self.adaptive:
            attention*=self.W
        out = a + attention
        return out

      #scaled dot pdt attention
    def dot_pdt_attention(self, q, k, v, dropout):
        #get n dimension size from [32,32,20,20]
        n = q.size(-1)
        k_t = k.transpose(-2,-1)

        scores = th.matmul(q, k_t)/np.sqrt(n)
        p_attn = F.softmax(scores,dim=-1)
        if dropout is not None:
          p_attn = dropout(p_attn)
        return th.matmul(p_attn, v)


class RvuAttn(nn.Module):
    def __init__(self, size, adaptive):
        super().__init__()
        #Approximator function for weight q,k,v
        self.w_q = nn.Conv2d(in_channels=size ,out_channels=size , kernel_size=1)
        self.w_k = nn.Conv2d(in_channels=size ,out_channels=size , kernel_size=1)
        self.w_v = nn.Conv2d(in_channels=size ,out_channels=size , kernel_size=1)

        self.adaptive = adaptive
        if adaptive:
            self.W = th.nn.Parameter(th.randn(1))
            self.W.requires_grad = True
    def forward(self, a, dropout=None):
        q = self.w_q(a).permute(0, 2, 3, 1)
        k = self.w_k(a).permute(0, 2, 3, 1)
        v = self.w_v(a).permute(0, 2, 3, 1)

        attention = self.dot_pdt_attention(q, k, v, dropout).permute(0, 3, 1, 2)
        if self.adaptive:
            attention*=self.W
        out = a + attention
        return out

    def dot_pdt_attention(self, q, k, v, dropout):
        #get n dimension size from [32,32,20,20]
        n = q.size(-1)
        k_t = k.transpose(-2,-1)

        scores = th.matmul(q, k_t)/np.sqrt(n)
        p_attn = F.softmax(scores,dim=-1)
        if dropout is not None:
          p_attn = dropout(p_attn)
        return th.matmul(p_attn, v)

class CrossAttn(nn.Module):
    def __init__(self, size, adaptive):
        super().__init__()
        #Approximator function for weight q,k,v
        self.w_q = nn.Conv2d(size, size, 1)
        self.w_k = nn.Conv2d(size, size, 1)
        self.w_v = nn.Conv2d(size, size, 1)

        self.adaptive = adaptive
        if adaptive:
            self.X = th.nn.Parameter(th.randn(1))
            self.X.requires_grad = True
            self.Y = th.nn.Parameter(th.randn(1))
            self.Y.requires_grad = True

    def forward(self, a, dropout=None):
        q = self.w_q(a)
        k = self.w_k(a)
        v = self.w_v(a)

        attentionL = self.dot_pdt_attention(q.permute(0, 2, 3, 1), k.permute(0, 2, 3, 1), v.permute(0, 2, 3, 1), dropout).permute(0, 3, 1, 2)

        attentionR = self.dot_pdt_attention(q.permute(0, 3, 2, 1), k.permute(0, 3, 2, 1), v.permute(0, 3, 2, 1), dropout).permute(0, 3, 2, 1)

        if self.adaptive:
            attentionL*=self.X
            attentionR*=self.Y
        return a + attentionL + attentionR

    def dot_pdt_attention(self, q, k, v, dropout):
        #get n dimension size from [32,32,20,20]
        n = q.size(-1)
        k_t = k.transpose(-2,-1)

        scores = th.matmul(q, k_t)/np.sqrt(n)
        p_attn = F.softmax(scores,dim=-1)
        if dropout is not None:
          p_attn = dropout(p_attn)
        return th.matmul(p_attn, v)


class xAttn(nn.Module):
    def __init__(self, size, adaptive, norm):
        super().__init__()
        #Approximator function for weight q,k,v
        self.w_q = nn.Conv2d(size, size, 1)
        self.w_k = nn.Conv2d(size, size, 1)
        self.w_v = nn.Conv2d(size, size, 1)

        self.norm = norm
        self.adaptive = adaptive
        if adaptive:
            self.X = th.nn.Parameter(th.randn(1))
            self.X.requires_grad = True
            self.Y = th.nn.Parameter(th.randn(1))
            self.Y.requires_grad = True

    def forward(self, a, dropout=None):
        q = self.w_q(a)
        k = self.w_k(a)
        v = self.w_v(a)

        attentionL = self.dot_pdt_attention(q.permute(0, 2, 3, 1), k.permute(0, 2, 3, 1), v.permute(0, 2, 3, 1), dropout).permute(0, 3, 1, 2)

        attentionR = self.dot_pdt_attention(q.permute(0, 3, 2, 1), k.permute(0, 3, 2, 1), v.permute(0, 3, 2, 1), dropout).permute(0, 3, 2, 1)

        attention = self.scaled_dot_pdt_attention(q, k, v, dropout)

        attentionL = th.matmul(attentionL,attention.transpose(-2,-1))
        attentionR = th.matmul(attentionR,attention.transpose(-2,-1))

        if self.norm: 
            attentionL = F.softmax(attentionL/np.sqrt(attentionL.size(-1)), dim=-1)
            attentionR = F.softmax(attentionR/np.sqrt(attentionR.size(-1)), dim=-1)
        if self.adaptive:
            attentionL*=self.X
            attentionR*=self.Y

        return a + attentionL + attentionR


    def dot_pdt_attention(self, q, k, v, dropout):
          #get n dimension size from [32,32,20,20]
        n = q.size(-1)
        k_t = k.transpose(-2,-1)

        scores = th.matmul(q, k_t)/np.sqrt(n)
        p_attn = F.softmax(scores,dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return th.matmul(p_attn, v)

    def scaled_dot_pdt_attention(self, q, k, v, dropout):
        attn = th.matmul(q, k.transpose(-2,-1))
        if self.norm:
            attn /= np.sqrt(q.size(-1))
            attn = F.softmax(attn,dim=-1)
        if dropout is not None:
            attn = dropout(attn)
        return th.matmul(attn, v)


def flatten_dict_observations(env: gym.Env) -> gym.Env:
    assert isinstance(env.observation_space, gym.spaces.Dict)
    try:
        return gym.wrappers.FlattenObservation(env)
    except AttributeError:
        keys = env.observation_space.spaces.keys()
        return gym.wrappers.FlattenDictWrapper(env, dict_keys=list(keys))


def get_wrapper_class(hyperparams: Dict[str, Any]) -> Optional[Callable[[gym.Env], gym.Env]]:
    """
    Get one or more Gym environment wrapper class specified as a hyper parameter
    "env_wrapper".
    e.g.
    env_wrapper: gym_minigrid.wrappers.FlatObsWrapper

    for multiple, specify a list:

    env_wrapper:
        - utils.wrappers.PlotActionWrapper
        - utils.wrappers.TimeFeatureWrapper


    :param hyperparams:
    :return: maybe a callable to wrap the environment
        with one or multiple gym.Wrapper
    """

    def get_module_name(wrapper_name):
        return ".".join(wrapper_name.split(".")[:-1])

    def get_class_name(wrapper_name):
        return wrapper_name.split(".")[-1]

    if "env_wrapper" in hyperparams.keys():
        wrapper_name = hyperparams.get("env_wrapper")

        if wrapper_name is None:
            return None

        if not isinstance(wrapper_name, list):
            wrapper_names = [wrapper_name]
        else:
            wrapper_names = wrapper_name

        wrapper_classes = []
        wrapper_kwargs = []
        # Handle multiple wrappers
        for wrapper_name in wrapper_names:
            # Handle keyword arguments
            if isinstance(wrapper_name, dict):
                assert len(wrapper_name) == 1, (
                    "You have an error in the formatting "
                    f"of your YAML file near {wrapper_name}. "
                    "You should check the indentation."
                )
                wrapper_dict = wrapper_name
                wrapper_name = list(wrapper_dict.keys())[0]
                kwargs = wrapper_dict[wrapper_name]
            else:
                kwargs = {}
            wrapper_module = importlib.import_module(get_module_name(wrapper_name))
            wrapper_class = getattr(wrapper_module, get_class_name(wrapper_name))
            wrapper_classes.append(wrapper_class)
            wrapper_kwargs.append(kwargs)

        def wrap_env(env: gym.Env) -> gym.Env:
            """
            :param env:
            :return:
            """
            for wrapper_class, kwargs in zip(wrapper_classes, wrapper_kwargs):
                env = wrapper_class(env, **kwargs)
            return env

        return wrap_env
    else:
        return None


def get_callback_list(hyperparams: Dict[str, Any]) -> List[BaseCallback]:
    """
    Get one or more Callback class specified as a hyper-parameter
    "callback".
    e.g.
    callback: stable_baselines3.common.callbacks.CheckpointCallback

    for multiple, specify a list:

    callback:
        - utils.callbacks.PlotActionWrapper
        - stable_baselines3.common.callbacks.CheckpointCallback

    :param hyperparams:
    :return:
    """

    def get_module_name(callback_name):
        return ".".join(callback_name.split(".")[:-1])

    def get_class_name(callback_name):
        return callback_name.split(".")[-1]

    callbacks = []

    if "callback" in hyperparams.keys():
        callback_name = hyperparams.get("callback")

        if callback_name is None:
            return callbacks

        if not isinstance(callback_name, list):
            callback_names = [callback_name]
        else:
            callback_names = callback_name

        # Handle multiple wrappers
        for callback_name in callback_names:
            # Handle keyword arguments
            if isinstance(callback_name, dict):
                assert len(callback_name) == 1, (
                    "You have an error in the formatting "
                    f"of your YAML file near {callback_name}. "
                    "You should check the indentation."
                )
                callback_dict = callback_name
                callback_name = list(callback_dict.keys())[0]
                kwargs = callback_dict[callback_name]
            else:
                kwargs = {}
            callback_module = importlib.import_module(get_module_name(callback_name))
            callback_class = getattr(callback_module, get_class_name(callback_name))
            callbacks.append(callback_class(**kwargs))

    return callbacks


def create_test_env(
    env_id: str,
    n_envs: int = 1,
    stats_path: Optional[str] = None,
    seed: int = 0,
    log_dir: Optional[str] = None,
    should_render: bool = True,
    hyperparams: Optional[Dict[str, Any]] = None,
    env_kwargs: Optional[Dict[str, Any]] = None,
) -> VecEnv:
    """
    Create environment for testing a trained agent

    :param env_id:
    :param n_envs: number of processes
    :param stats_path: path to folder containing saved running averaged
    :param seed: Seed for random number generator
    :param log_dir: Where to log rewards
    :param should_render: For Pybullet env, display the GUI
    :param hyperparams: Additional hyperparams (ex: n_stack)
    :param env_kwargs: Optional keyword argument to pass to the env constructor
    :return:
    """
    # Create the environment and wrap it if necessary
    env_wrapper = get_wrapper_class(hyperparams)

    hyperparams = {} if hyperparams is None else hyperparams

    if "env_wrapper" in hyperparams.keys():
        del hyperparams["env_wrapper"]

    vec_env_kwargs = {}
    vec_env_cls = DummyVecEnv
    if n_envs > 1 or "Bullet" in env_id:
        # HACK: force SubprocVecEnv for Bullet env
        # as Pybullet envs does not follow gym.render() interface
        vec_env_cls = SubprocVecEnv
        # start_method = 'spawn' for thread safe

    env = make_vec_env(
        env_id,
        n_envs=n_envs,
        monitor_dir=log_dir,
        seed=seed,
        wrapper_class=env_wrapper,
        env_kwargs=env_kwargs,
        vec_env_cls=vec_env_cls,
        vec_env_kwargs=vec_env_kwargs,
    )

    # Load saved stats for normalizing input and rewards
    # And optionally stack frames
    if stats_path is not None:
        if hyperparams["normalize"]:
            print("Loading running average")
            print(f"with params: {hyperparams['normalize_kwargs']}")
            path_ = os.path.join(stats_path, "vecnormalize.pkl")
            if os.path.exists(path_):
                env = VecNormalize.load(path_, env)
                # Deactivate training and reward normalization
                env.training = False
                env.norm_reward = False
            else:
                raise ValueError(f"VecNormalize stats {path_} not found")

        n_stack = hyperparams.get("frame_stack", 0)
        if n_stack > 0:
            print(f"Stacking {n_stack} frames")
            env = VecFrameStack(env, n_stack)
    return env


def linear_schedule(initial_value: Union[float, str]) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: (float or str)
    :return: (function)
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress_remaining: (float)
        :return: (float)
        """
        return progress_remaining * initial_value

    return func


def get_trained_models(log_folder: str) -> Dict[str, Tuple[str, str]]:
    """
    :param log_folder: (str) Root log folder
    :return: (Dict[str, Tuple[str, str]]) Dict representing the trained agent
    """
    trained_models = {}
    for algo in os.listdir(log_folder):
        if not os.path.isdir(os.path.join(log_folder, algo)):
            continue
        for env_id in os.listdir(os.path.join(log_folder, algo)):
            # Retrieve env name
            env_id = env_id.split("_")[0]
            trained_models[f"{algo}-{env_id}"] = (algo, env_id)
    return trained_models


def get_latest_run_id(log_path: str, env_id: str) -> int:
    """
    Returns the latest run number for the given log name and log path,
    by finding the greatest number in the directories.

    :param log_path: path to log folder
    :param env_id:
    :return: latest run number
    """
    max_run_id = 0
    for path in glob.glob(log_path + f"/{env_id}_[0-9]*"):
        file_name = path.split("/")[-1]
        ext = file_name.split("_")[-1]
        if env_id == "_".join(file_name.split("_")[:-1]) and ext.isdigit() and int(ext) > max_run_id:
            max_run_id = int(ext)
    return max_run_id


def get_saved_hyperparams(stats_path: str, norm_reward: bool = False, test_mode: bool = False) -> Tuple[Dict[str, Any], str]:
    """
    :param stats_path:
    :param norm_reward:
    :param test_mode:
    :return:
    """
    hyperparams = {}
    if not os.path.isdir(stats_path):
        stats_path = None
    else:
        config_file = os.path.join(stats_path, "config.yml")
        if os.path.isfile(config_file):
            # Load saved hyperparameters
            with open(os.path.join(stats_path, "config.yml"), "r") as f:
                hyperparams = yaml.load(f, Loader=yaml.UnsafeLoader)  # pytype: disable=module-attr
            hyperparams["normalize"] = hyperparams.get("normalize", False)
        else:
            obs_rms_path = os.path.join(stats_path, "obs_rms.pkl")
            hyperparams["normalize"] = os.path.isfile(obs_rms_path)

        # Load normalization params
        if hyperparams["normalize"]:
            if isinstance(hyperparams["normalize"], str):
                normalize_kwargs = eval(hyperparams["normalize"])
                if test_mode:
                    normalize_kwargs["norm_reward"] = norm_reward
            else:
                normalize_kwargs = {"norm_obs": hyperparams["normalize"], "norm_reward": norm_reward}
            hyperparams["normalize_kwargs"] = normalize_kwargs
    return hyperparams, stats_path


class StoreDict(argparse.Action):
    """
    Custom argparse action for storing dict.

    In: args1:0.0 args2:"dict(a=1)"
    Out: {'args1': 0.0, arg2: dict(a=1)}
    """

    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        self._nargs = nargs
        super(StoreDict, self).__init__(option_strings, dest, nargs=nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        arg_dict = {}
        for arguments in values:
            key = arguments.split(":")[0]
            value = ":".join(arguments.split(":")[1:])
            # Evaluate the string as python code
            arg_dict[key] = eval(value)
        setattr(namespace, self.dest, arg_dict)
