import math
import numpy as np
import torch
from torch import optim
from torch import nn
from torch.distributions import Categorical
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import gym

from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.env_util import make_atari_env

env_seed = 420
save_dir = "PPORvuAttn/seed"+str(env_seed)+"/"

import os
os.makedirs(save_dir, exist_ok=True)

env = make_atari_env('DemonAttackNoFrameskip-v4', n_envs=64, monitor_dir=save_dir,
                     seed=env_seed, wrapper_kwargs=dict(noop_max=30,frame_skip=4,screen_size=84, terminal_on_life_loss=True, clip_reward=True))
# Frame-stacking with 4 frames
vec_env = VecFrameStack(env, n_stack=4)

eval_env = make_atari_env('DemonAttackNoFrameskip-v4', n_envs=1, monitor_dir=save_dir,
                     seed=env_seed, wrapper_kwargs=dict(noop_max=30,frame_skip=4,screen_size=84, terminal_on_life_loss=True, clip_reward=True))
eval_env = VecFrameStack(eval_env, n_stack=4)

import base64
from pathlib import Path

from IPython import display as ipythondisplay

class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            MultiHeadAttn(size=32),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

class MultiHeadAttn(nn.Module):
  def __init__(self, size):
    super().__init__()
    #Approximator function for weight q,k,v
    self.w_q = nn.Conv2d(in_channels=size ,out_channels=size , kernel_size=1)
    self.w_k = nn.Conv2d(in_channels=size ,out_channels=size , kernel_size=1)
    self.w_v = nn.Conv2d(in_channels=size ,out_channels=size , kernel_size=1)
    #self.out = nn.Conv2d(in_channels=size ,out_channels=size , kernel_size=1)

  def forward(self, a, dropout=None):
    q = self.w_q(a).permute(0, 2, 3, 1)
    k = self.w_k(a).permute(0, 2, 3, 1)
    v = self.w_v(a).permute(0, 2, 3, 1)

    attention = self.dot_pdt_attention(q, k, v, dropout).permute(0, 3, 1, 2)
    #out = F.relu(self.out(attention))
    out = a + attention
    return out

  #scaled dot pdt attention
  def dot_pdt_attention(self, q, k, v, dropout):
        attn = torch.matmul(q, k.transpose(2, 3))
        output = torch.matmul(attn, v)
        return output

policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=512),
)

eval_callback = EvalCallback(eval_env, best_model_save_path= save_dir+'/logs/',
                             log_path=save_dir+'/logs/', eval_freq=100000,
                             deterministic=True, render=False)

model = PPO("CnnPolicy", vec_env, verbose=1, policy_kwargs=policy_kwargs,
            tensorboard_log=(save_dir+'runs')).learn(total_timesteps=10000000, callback=eval_callback)

# Train the agent for 10000 steps
model.save(save_dir + "/PPO")

def show_videos(video_path='', prefix=''):
  """
  Taken from https://github.com/eleurent/highway-env

  :param video_path: (str) Path to the folder containing videos
  :param prefix: (str) Filter the video, showing only the only starting with this prefix
  """
  html = []
  for mp4 in Path(video_path).glob("{}*.mp4".format(prefix)):
      video_b64 = base64.b64encode(mp4.read_bytes())
      html.append('''<video alt="{}" autoplay
                    loop controls style="height: 400px;">
                    <source src="data:video/mp4;base64,{}" type="video/mp4" />
                </video>'''.format(mp4, video_b64.decode('ascii')))
  ipythondisplay.display(ipythondisplay.HTML(data="<br>".join(html)))

from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv

def record_video(env_id, model, video_length=500, prefix='', video_folder='videos/'):
  """
  :param env_id: (str)
  :param model: (RL model)
  :param video_length: (int)
  :param prefix: (str)
  :param video_folder: (str)
  """
  eval_env = make_atari_env('DemonAttackNoFrameskip-v4', n_envs=4,
                     seed=env_seed, wrapper_kwargs=dict(noop_max=30,frame_skip=4,screen_size=84, terminal_on_life_loss=True, clip_reward=True))
  # Frame-stacking with 4 frames
  eval_env = VecFrameStack(env, n_stack=4)
  # Start the video at step=0 and record 500 steps
  eval_env = VecVideoRecorder(eval_env, video_folder=video_folder,
                              record_video_trigger=lambda step: step == 0, video_length=video_length,
                              name_prefix=prefix)

  obs = eval_env.reset()
  for _ in range(video_length):
    action, _ = model.predict(obs)
    obs, _, _, _ = eval_env.step(action)

  # Close the video recorder
  eval_env.close()

record_video('DemonAttackNoFrameskip-v4', model, video_length=5000, prefix='ppo-demonattack', video_folder=save_dir)
