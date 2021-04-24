import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal
import math

class ActorCriticNet(nn.Module):
    def __init__(self, shape, ac_s, attn_type: str = 'Attn',adaptive: bool = False):
        super().__init__()
        if attn_type == 'Attn':
            self.layer = MultiHeadAttn(size=32, adaptive=adaptive)
        elif attn_type == 'RvuAttn':
            self.layer = RvuAttn(size=32, adaptive=adaptive)
        elif attn_type == 'CrossAttn':
            self.layer = CrossAttn(size=32, adaptive=adaptive)
        elif attn_type == 'xAttn':
            self.layer = xAttn(size=32, adaptive=adaptive)
        else:
            self.layer = None
        self.c1 = nn.Conv2d(shape[0], 32, 8, stride=4)
        self.c2 = nn.Conv2d(32, 64, 4, stride=2)
        self.c3 = nn.Conv2d(64, 64, 3, stride=1)
        self.conv_out = self._get_conv_out(shape)

        self.l1 = nn.Linear(self.conv_out, 512)
        self.critic = nn.Linear(512, 1)
        self.actor = nn.Linear(512, ac_s)

    def cnn_layer(self, x):
        h = F.relu(self.c1(x))
        h = self.layer(h)
        h = F.relu(self.c2(h))
        h = F.relu(self.c3(h))
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
            self.W = torch.nn.Parameter(torch.randn(1))
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

        scores = torch.matmul(q, k_t)/np.sqrt(n)
        p_attn = F.softmax(scores,dim=-1)
        if dropout is not None:
          p_attn = dropout(p_attn)
        return torch.matmul(p_attn, v)


class RvuAttn(nn.Module):
    def __init__(self, size, adaptive):
        super().__init__()
        #Approximator function for weight q,k,v
        self.w_q = nn.Conv2d(in_channels=size ,out_channels=size , kernel_size=1)
        self.w_k = nn.Conv2d(in_channels=size ,out_channels=size , kernel_size=1)
        self.w_v = nn.Conv2d(in_channels=size ,out_channels=size , kernel_size=1)
        self.adaptive = adaptive
        if adaptive:
            self.W = torch.nn.Parameter(torch.randn(1))
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
        attn = torch.matmul(q, k.transpose(-2,-1))
        attn /= np.sqrt(q.size(-1))
        attn = F.softmax(attn,dim=-1)
        if dropout is not None:
            attn = dropout(attn)
        return torch.matmul(attn, v)

class CrossAttn(nn.Module):
    def __init__(self, size, adaptive):
        super().__init__()
        #Approximator function for weight q,k,v
        self.w_q = nn.Conv2d(size, size, 1)
        self.w_k = nn.Conv2d(size, size, 1)
        self.w_v = nn.Conv2d(size, size, 1)

        self.adaptive = adaptive
        if adaptive:
            self.X = torch.nn.Parameter(torch.randn(1))
            self.X.requires_grad = True
            self.Y = torch.nn.Parameter(torch.randn(1))
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
        attn = torch.matmul(q, k.transpose(-2,-1))
        attn /= np.sqrt(q.size(-1))
        attn = F.softmax(attn,dim=-1)
        if dropout is not None:
            attn = dropout(attn)
        return torch.matmul(attn, v)


class xAttn(nn.Module):
    def __init__(self, size, adaptive):
        super().__init__()
        #Approximator function for weight q,k,v
        self.w_q = nn.Conv2d(size, size, 1)
        self.w_k = nn.Conv2d(size, size, 1)
        self.w_v = nn.Conv2d(size, size, 1)

        self.adaptive = adaptive
        if adaptive:
            self.X = torch.nn.Parameter(torch.randn(1))
            self.X.requires_grad = True
            self.Y = torch.nn.Parameter(torch.randn(1))
            self.Y.requires_grad = True

    def forward(self, a, dropout=None):
        q = self.w_q(a)
        k = self.w_k(a)
        v = self.w_v(a)

        attentionL = self.dot_pdt_attention(q.permute(0, 2, 3, 1), k.permute(0, 2, 3, 1), v.permute(0, 2, 3, 1), dropout).permute(0, 3, 1, 2)

        attentionR = self.dot_pdt_attention(q.permute(0, 3, 2, 1), k.permute(0, 3, 2, 1), v.permute(0, 3, 2, 1), dropout).permute(0, 3, 2, 1)

        attention = self.scaled_dot_pdt_attention(q, k, v, dropout)

        attentionL = torch.matmul(attentionL,attention.transpose(-2,-1))
        attentionR = torch.matmul(attentionR,attention.transpose(-2,-1))

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

        p_attn = torch.matmul(q, k_t)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, v)

    def scaled_dot_pdt_attention(self, q, k, v, dropout):
        attn = torch.matmul(q, k.transpose(-2,-1))
        if dropout is not None:
            attn = dropout(attn)
        return torch.matmul(attn, v)
