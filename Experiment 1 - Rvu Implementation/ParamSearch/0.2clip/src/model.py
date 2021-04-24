import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal
import math

class ActorCriticNet(nn.Module):
    def __init__(self, shape, ac_s):
        super().__init__()
        self.c1 = nn.Conv2d(shape[0], 32, 8, stride=4)
        self.attention_layer = MultiHeadAttention(32)
        self.c2 = nn.Conv2d(32, 64, 4, stride=2)
        self.attention_layer2 = MultiHeadAttention(64)
        self.c3 = nn.Conv2d(64, 64, 3, stride=1)
        self.attention_layer3 = MultiHeadAttention(64)
        self.conv_out = self._get_conv_out(shape)

        self.l1 = nn.Linear(self.conv_out, 512)
        self.critic = nn.Linear(512, 1)
        self.actor = nn.Linear(512, ac_s)

    def cnn_layer(self, x):
        h = F.relu(self.c1(x))
        #h = self.attention_layer(h, h, h)
        h = F.relu(self.c2(h))
        #h = self.attention_layer2(h,h,h)
        h = F.relu(self.c3(h))
        #h = self.attention_layer3(h,h,h)
        return h

    def shared_layer(self, x):
        h = self.cnn_layer(x)
        h = h.reshape(-1).view(-1, self.conv_out)
        h = F.relu(self.l1(h))
        return h

    def forward(self, x):
        h = self.shared_layer(x)
        actor_logits = self.actor(h)
        values = self.critic(h)
        prob = F.softmax(actor_logits, dim=-1)
        acts = prob.multinomial(1)
        return actor_logits, values, acts

    def _get_conv_out(self, shape):
        x = torch.zeros(1, *shape)
        h = self.cnn_layer(x)
        return int(np.prod(h.size()))

class MultiHeadAttention(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.w_qs = nn.Conv2d(size, size, 1)
        self.w_ks = nn.Conv2d(size, size, 1)
        self.w_vs = nn.Conv2d(size, size, 1)
        #self.W = torch.nn.Parameter(torch.randn(1))
        #self.W.requires_grad = True
        #self.Y = torch.nn.Parameter(torch.randn(1))
        #self.Y.requires_grad = True

        self.attention = ScaledDotProductAttention()

    def forward(self, q, k, v):

        residual = q
        q = self.w_qs(q).permute(0, 2, 3, 1)
        k = self.w_ks(k).permute(0, 2, 3, 1)
        v = self.w_vs(v).permute(0, 2, 3, 1)

        attention = self.attention(q, k, v).permute(0, 3, 1, 2)

        out = attention + residual #self.W*attention + self.Y*residual
        return out


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v):
        attn = torch.matmul(q, k.transpose(2, 3))
        output = torch.matmul(attn, v)

        return output
"""
        n = q.size(-1)
        k_t = k.transpose(-2,-1)

        scores = torch.matmul(q, k_t)/math.sqrt(n)
        p_attn = F.softmax(scores,dim=-1)
        return torch.matmul(p_attn, v)
"""
