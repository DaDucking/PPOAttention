import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.distributions import Categorical

from .multiprocess import SubprocVecEnv
from .model import ActorCriticNet
from .wrappers import make_env_function, make_env_with_wrappers
import torchcam
from torchvision.transforms.functional import normalize, resize, to_tensor, to_pil_image
from torchcam import cams
from torchcam.utils import overlay_mask
import matplotlib
class PPO:
    def __init__(
        self,
        env_name,
        max_epochs,
        n_envs,
        n_steps,
        batch_size,
        writer,
        epsilon=0.2,
        gamma=0.99,
        lambda_=0.95,
        v_loss_coef=0.5,
        entropy_coef=0.001,
        max_grad_norm=0.5,
        lr=0.0003,
        ppo_epochs=4,
        train_seed=142,
        cuda = "cuda:0",
        attn_type = "NoAttn",
        adaptive = False,
    ):
        self.env_name = env_name
        self.envs = SubprocVecEnv([make_env_function(env_name, train_seed) for _ in range(n_envs)])
        self.env = make_env_with_wrappers(env_name, train_seed%13*train_seed//2)
        self.attn_type = attn_type
        self.obs_space = self.envs.observation_space.shape
        action_space = self.envs.action_space.n
        self.model = ActorCriticNet(self.obs_space, action_space, attn_type, adaptive)
        self.device = cuda if torch.cuda.is_available() else "cpu:0"
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.writer = writer

        self.max_epochs = max_epochs
        self.n_envs = n_envs
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.batch_num = self.n_steps * self.n_envs // self.batch_size

        self.gamma = gamma
        self.epsilon = epsilon
        self.lambda_ = lambda_
        self.v_loss_coef = v_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.cuda = cuda

    def load(self):
        model_dict = self.model.state_dict()
        pretrained_dict = torch.load("model.pt")

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict }
        if self.env_name != "PongNoFrameskip-v4":
            pretrained_dict['actor.weight'] = model_dict['actor.weight']
            pretrained_dict['actor.bias'] = model_dict['actor.bias']
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(pretrained_dict, self.device)
        self.model.to(self.device)

    def Gradcam(self):
        states = torch.zeros([self.n_steps + 1, self.n_env, *self.obs_space]).to(
            self.device
        )
        states[0] = torch.from_numpy(self.env.reset())
        matplotlib.image.imsave('name.png', states[0].to("cpu").numpy()[0])
        for t in range(3):
            pass

    def record_video(self, video_length=100):
        rec_env = Monitor(self.env, './video', force=True)

        obs = torch.from_numpy(rec_env.reset())
        for _ in range(video_length):
            _, _,action = self.model(obs)
            obs, _, _, _ = rec_env.step(action)
            obs = torch.from_numpy(obs)
        env.close()

    def train(self):
        states, actions, rewards, advantages, returns, masks = self._rollout()

        best_score = -22

        for epoch in range(self.max_epochs):
            self._update(states, actions, rewards, advantages, returns, masks, epoch)
            del states
            states, actions, rewards, advantages, returns, masks = self._rollout()

            print("epoch:", epoch + 1, end=", ")
            if (epoch + 1) % 1 == 0:
                score = np.mean([self._test_env() for _ in range(5)])
                test_score = np.mean([self._test_deterministic_env() for _ in range(5)])

                if test_score > best_score:
                    best_score = test_score
                    torch.save(self.model.state_dict(), "model.pt")
                    print("saved best model with", end=" ")
                if (epoch + 1) % 10 == 0:
                    model_name = self.attn_type+"Epoch"+str(epoch+1)+"model.pt"
                    torch.save(self.model.state_dict(), model_name)

                self.writer.add_scalar("Score", score, epoch + 1)
                print("score:", score, end=", ")
                print("   test_score:", test_score)
                self.writer.add_scalar("Test_Score", test_score, epoch + 1)
                #self.writer.add_scalar("X", self.model.attention_layer.W.cpu().detach().item(), epoch + 1)
                #self.writer.add_scalar("Y", self.model.attention_layer.Y.cpu().detach().item(), epoch + 1)

            if best_score >= 10000:
                print("Finished training!")
                break

    def _rollout(self):
        states = torch.zeros([self.n_steps + 1, self.n_envs, *self.obs_space]).to(
            self.device
        )
        masks = np.ones([self.n_steps + 1, self.n_envs, 1], dtype=np.float32)
        rewards = np.zeros([self.n_steps, self.n_envs, 1], dtype=np.float32)
        actions = np.zeros([self.n_steps, self.n_envs, 1], dtype=np.int32)
        values = np.zeros([self.n_steps + 1, self.n_envs, 1], dtype=np.float32)

        states[0] = torch.from_numpy(self.envs.reset())
        masks[0] = 0.0

        with torch.no_grad():
            for t in range(self.n_steps):
                _, vals, acts = self.model(states[t])

                actions[t] = acts.to("cpu").numpy()
                values[t] = vals.to("cpu").numpy()
                states_np, rewards[t, :, 0], dones, _ = self.envs.step(actions[t])
                masks[t][dones] = 0
                states[t + 1] = torch.from_numpy(states_np)

            _, last_val, _ = self.model(states[-1])
            values[-1] = last_val.to("cpu").numpy()
            advantages = np.zeros([self.n_steps, self.n_envs, 1], dtype=np.float32)

            returns = np.zeros([self.n_steps, self.n_envs, 1], dtype=np.float32)
            returns[-1] = rewards[-1] + self.gamma * masks[-1] * values[-1]
            for t in reversed(range(self.n_steps - 1)):
                returns[t] = rewards[t] + self.gamma * masks[t + 1] * returns[t + 1]

            gae = 0
            for t in reversed(range(self.n_steps)):
                delta = -values[t] + rewards[t] + self.gamma * values[t + 1]
                gae = delta + self.gamma * self.lambda_ * masks[t] * gae
                advantages[t] = gae

        states = states[:-1].view(-1, *self.obs_space)
        actions = torch.from_numpy(actions).long().view(-1, 1)
        returns = torch.from_numpy(returns).view(-1, 1)
        advantages = torch.from_numpy(advantages).view(-1, 1)

        del values

        return states, actions, rewards, advantages, returns, masks

    def _update(self, states, actions, rewards, advantages, returns, masks, epoch):
        old_model = copy.deepcopy(self.model)

        policy_losses = np.array([])
        entropies = np.array([])
        value_losses = np.array([])
        losses = np.array([])

        for _ in range(self.ppo_epochs):
            rand_list = (
                torch.randperm(self.batch_num * self.batch_size)
                .view(-1, self.batch_size)
                .tolist()
            )

            for ind in rand_list:
                batch = states[ind]
                actor_logits, vals, _ = self.model(batch)
                log_probs = F.log_softmax(actor_logits, dim=1)
                with torch.no_grad():
                    old_actor_logits, _, _ = old_model(batch)
                    old_log_probs = F.log_softmax(old_actor_logits, dim=1)

                adv = advantages[ind].to(self.device)
                advs = advantages.to(self.device)
                adv = (adv - advs.mean())/(advs.std() + 1e-8)

                A = returns[ind].to(self.device) - vals

                action = actions[ind].to(self.device)

                old_log_probs = old_log_probs.gather(1, action)
                log_probs = log_probs.gather(1, action)

                r = (log_probs - old_log_probs).exp()

                clip = r.clamp(min=1 - self.epsilon, max=1 + self.epsilon)
                L, _ = torch.stack([r * adv.detach(), clip * adv.detach()]).min(0)
                v_l = A.pow(2).mean()
                L = L.mean()

                entropy = Categorical(F.softmax(actor_logits, dim=1)).entropy().mean()

                loss = -L + self.v_loss_coef * v_l - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                policy_losses = np.append(policy_losses, L.cpu().detach().numpy())
                value_losses = np.append(value_losses, v_l.cpu().detach().numpy())
                losses = np.append(losses, loss.cpu().detach().numpy())
                entropies = np.append(entropies, entropy.cpu().detach().numpy())

        policy_loss = policy_losses.mean()
        value_loss = value_losses.mean()
        loss = losses.mean()
        entropy = entropies.mean()

        self.writer.add_scalar("PolicyLoss", policy_loss, epoch + 1)
        self.writer.add_scalar("ValueLoss", value_loss, epoch + 1)
        self.writer.add_scalar("Loss", loss, epoch + 1)
        self.writer.add_scalar("Entropy", entropy, epoch + 1)

        del states, actions, rewards, advantages, returns, masks

    def _test_env(self):

        state = self.env.reset()
        done = False
        total_reward = 0

        while not done:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.cuda)
            _, _, action = self.model(state)
            next_state, reward, done, _ = self.env.step(action.to("cpu"))

            state = next_state
            total_reward += reward
        return total_reward

    def _test_deterministic_env(self):
        state = self.env.reset()
        done = False
        total_reward = 0

        while not done:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.cuda)
            actor_logits, _, _ = self.model(state)
            action = torch.argmax(actor_logits, dim=1)
            next_state, reward, done, _ = self.env.step(action.to("cpu"))

            state = next_state
            total_reward += reward
        return total_reward

    def eval(self, num_of_games):
        for _ in range(num_of_games):
            self.model.load_state_dict(torch.load("model.pt"))
            self.model.eval()

            state = self.env.reset()
            done = False

            while not done:
                state = torch.FloatTensor(state).unsqueeze(0).to(self.cuda)
                action = self.model.act(state).to("cpu")
                self.env.render(mode="human")
                state, _, done, _ = self.env.step(action)
