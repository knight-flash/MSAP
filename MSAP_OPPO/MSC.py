

import argparse

parser = argparse.ArgumentParser()

import torch

parser = argparse.ArgumentParser(description="Experiment Info and Setings, Model Hyperparameters")
parser.add_argument("--lambda_cls", type=float, default=1)
parser.add_argument("--lambda_sc", type=float, default=2)
parser.add_argument("--lambda_st", type=float, default=0.2)
parser.add_argument("--lambda_cos_loss", type=float, default=2)
# Experiment Info
parser.add_argument("--characteristic", '-c', type=str, default="")
parser.add_argument("--data", type=str, default='Sleep-edf')
parser.add_argument("--data_type", type=str, default='epoch')
parser.add_argument("--scheme", type=str, default='M_M')
parser.add_argument("--loss_weight", type=int, default=1)
parser.add_argument("--lstm_layers", type=int, default=1)
parser.add_argument("--cos_loss", type=int, default=1)
parser.add_argument("--mha", type=int, default=1)
parser.add_argument("--mha_length", type=int, default=8)
parser.add_argument("--mha_head", type=int, default=2)
parser.add_argument("--mass_ch", type=str, default='eeg_f4-ler')
parser.add_argument("--downsample", type=int, default=100)
# Experiment Hyperparameters
parser.add_argument("--epoch", type=int, default=150)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--wd", type=float, default=1e-3)
parser.add_argument("--batch", type=int, default=64)
parser.add_argument("--early_stop", type=int, default=50)
parser.add_argument("--dropout", type=int, default=0.5)
parser.add_argument("--scheduler", type=int, default=0)
parser.add_argument("--stride", type=str, default=2)
parser.add_argument("--preprocess", type=str, default='robustscale')
# Model Hyperparameters
parser.add_argument("--seq_length", type=int, default=8)
# GPU
parser.add_argument("--GPU", type=bool, default=True)
parser.add_argument("--gpu_idx", type=int, default=-1)
# Experiment Sbj
parser.add_argument("--range_start", type=int, default=0)
parser.add_argument("--range_end", type=int, default=31)
args = parser.parse_args(args=[])
# %%
import torch.nn as nn
import torch
#from retention import MultiScaleRetention
import math
import torch
import torch.nn as nn
device = torch.device("cuda")
class EfficientChannelAttention(nn.Module):           # Efficient Channel Attention module
    def __init__(self, c, b=1, gamma=2):
        super(EfficientChannelAttention, self).__init__()
        t = int(abs((math.log(c, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k, padding=int(k/2), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = x
        x = self.avg_pool(x)
        x = self.conv1(x.transpose(-1, -2)).transpose(-1, -2)
        out = self.sigmoid(x)
        return out+y
def fixed_pos_embedding(x):
    seq_len, dim = x.shape
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim) / dim))
    sinusoid_inp = (
        torch.einsum("i , j -> i j", torch.arange(0, seq_len, dtype=torch.float), inv_freq).to(x)
    )
    return torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)

def rotate_every_two(x):
    x1 = x[:, :, ::2]
    x2 = x[:, :, 1::2]

    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)

    # in einsum notation: rearrange(x, '... d j -> ... (d j)')\

def duplicate_interleave(m):
    """
    A simple version of `torch.repeat_interleave` for duplicating a matrix while interleaving the copy.
    """
    dim0 = m.shape[0]
    m = m.view(-1, 1)  # flatten the matrix
    m = m.repeat(1, 2)  # repeat all elements into the 2nd dimension
    m = m.view(dim0, -1)  # reshape into a matrix, interleaving the copy
    return m

def apply_rotary_pos_emb(x, sin, cos, scale=1):
    sin, cos = map(lambda t: duplicate_interleave(t * scale), (sin, cos))
    # einsum notation for lambda t: repeat(t[offset:x.shape[1]+offset,:], "n d -> () n () (d j)", j=2)
    return (x * cos) + (rotate_every_two(x) * sin)

class XPOS(nn.Module):
    def __init__(
            self, head_dim, scale_base=512
    ):
        super().__init__()
        self.head_dim = head_dim
        self.scale_base = scale_base
        self.register_buffer(
            "scale", (torch.arange(0, head_dim, 2) + 0.4 * head_dim) / (1.4 * head_dim)
        )

    def forward(self, x, offset=0, downscale=False):
        length = x.shape[1]
        min_pos = 0
        max_pos = length + offset + min_pos
        scale = self.scale ** torch.arange(min_pos, max_pos, 1).to(self.scale).div(self.scale_base)[:, None]
        sin, cos = fixed_pos_embedding(scale)

        if scale.shape[0] > length:
            scale = scale[-length:]
            sin = sin[-length:]
            cos = cos[-length:]

        if downscale:
            scale = 1 / scale

        x = apply_rotary_pos_emb(x, sin, cos, scale)
        return x

    def forward_reverse(self, x, offset=0, downscale=False):
        length = x.shape[1]
        min_pos = -(length + offset) // 2
        max_pos = length + offset + min_pos
        scale = self.scale ** torch.arange(min_pos, max_pos, 1).to(self.scale).div(self.scale_base)[:, None]
        sin, cos = fixed_pos_embedding(scale)

        if scale.shape[0] > length:
            scale = scale[-length:]
            sin = sin[-length:]
            cos = cos[-length:]

        if downscale:
            scale = 1 / scale

        x = apply_rotary_pos_emb(x, -sin, cos, scale)
        return x
class SimpleRetention(nn.Module):
    def __init__(self, hidden_size, gamma, head_size=None, double_v_dim=False):
        """
        Simple retention mechanism based on the paper
        "Retentive Network: A Successor to Transformer for Large Language Models"[https://arxiv.org/pdf/2307.08621.pdf]
        """
        super(SimpleRetention, self).__init__()

        self.hidden_size = hidden_size
        if head_size is None:
            head_size = hidden_size
        self.head_size = head_size

        self.v_dim = head_size * 2 if double_v_dim else head_size
        self.gamma = gamma
        self.W_Q = nn.Parameter(torch.randn(hidden_size, head_size) / hidden_size)
        self.W_K = nn.Parameter(torch.randn(hidden_size, head_size) / hidden_size)
        self.W_V = nn.Parameter(torch.randn(hidden_size, self.v_dim) / hidden_size)

        self.xpos = XPOS(head_size)

    def forward(self, X):
        """
        Parallel (default) representation of the retention mechanism.
        X: (batch_size, sequence_length, hidden_size)
        """
        sequence_length = X.shape[1]
        D = self._get_D(sequence_length)

        Q = (X @ self.W_Q)
        K = (X @ self.W_K)

        Q = self.xpos(Q)
        K = self.xpos(K, downscale=True)

        V = X @ self.W_V
        Q, K, V ,D= Q.to(device), K.to(device), V.to(device),D.to(device)

        ret = (Q @ K.permute(0, 2, 1)) * D.unsqueeze(0)

        return ret @ V

    def forward_recurrent(self, x_n, s_n_1, n):
        """
        Recurrent representation of the retention mechanism.
        x_n: (batch_size, 1, hidden_size)
        s_n_1: (batch_size, hidden_size, v_dim)
        """

        Q = (x_n @ self.W_Q)
        K = (x_n @ self.W_K)

        Q = self.xpos(Q, n+1)
        K = self.xpos(K, n+1, downscale=True)

        V = x_n @ self.W_V

        # K: (batch_size, 1, hidden_size)
        # V: (batch_size, 1, v_dim)
        # s_n = gamma * s_n_1 + K^T @ V

        s_n = self.gamma * s_n_1 + (K.transpose(-1, -2) @ V)

        return (Q @ s_n), s_n

    def forward_chunkwise(self, x_i, r_i_1, i):
        """
        Chunkwise representation of the retention mechanism.
        x_i: (batch_size, chunk_size, hidden_size)
        r_i_1: (batch_size, hidden_size, v_dim)
        """
        batch, chunk_size, _ = x_i.shape
        D = self._get_D(chunk_size)
        x_i = x_i.to(self.W_Q.device) 
        self.W_Q = self.W_Q.to(x_i.device)
        self.W_K = self.W_K.to(x_i.device)
        Q = (x_i @ self.W_Q)
        K = (x_i @ self.W_K)

        Q = self.xpos(Q, i * chunk_size)
        K = self.xpos(K, i * chunk_size, downscale=True)
        
        V = x_i @ self.W_V

        #print(r_i_1.shape)
        r_i_1 = r_i_1[:K.shape[0]]  
        r_i =(K.transpose(-1, -2) @ (V * D[-1].view(1, chunk_size, 1))) + (self.gamma ** chunk_size) * r_i_1

        inner_chunk = ((Q @ K.transpose(-1, -2)) * D.unsqueeze(0)) @ V

        #e[i,j] = gamma ** (i+1)
        e = torch.zeros(batch, chunk_size, 1)

        for _i in range(chunk_size):
            e[:, _i, :] = self.gamma ** (_i + 1)

        cross_chunk = (Q @ r_i_1) * e

        return inner_chunk + cross_chunk, r_i

    def _get_D(self, sequence_length):
        n = torch.arange(sequence_length).unsqueeze(1)
        m = torch.arange(sequence_length).unsqueeze(0)

        # Broadcast self.gamma ** (n - m) with appropriate masking to set values where n < m to 0
        D = (self.gamma ** (n - m)) * (n >= m).float()  #this results in some NaN when n is much larger than m
        # fill the NaN with 0
        D[D != D] = 0

        return D



class MultiScaleRetention(nn.Module):
    def __init__(self, hidden_size, heads, double_v_dim=False):
        """
        Multi-scale retention mechanism based on the paper
        "Retentive Network: A Successor to Transformer for Large Language Models"[https://arxiv.org/pdf/2307.08621.pdf]
        """
        super(MultiScaleRetention, self).__init__()
        self.hidden_size = hidden_size
        self.v_dim = hidden_size * 2 if double_v_dim else hidden_size
        self.heads = heads
        assert hidden_size % heads == 0, "hidden_size must be divisible by heads"
        self.head_size = hidden_size // heads
        self.head_v_dim = hidden_size * 2 if double_v_dim else hidden_size

        self.gammas = (1 - torch.exp(torch.linspace(math.log(1/32), math.log(1/512), heads))).detach().cpu().tolist()

        #self.swish = lambda x: x * torch.sigmoid(x)
        self.W_G = nn.Parameter(torch.randn(hidden_size, self.v_dim) / hidden_size)
        self.W_O = nn.Parameter(torch.randn(self.v_dim, hidden_size) / hidden_size)
        self.group_norm = nn.GroupNorm(heads, self.v_dim)

        self.retentions = nn.ModuleList([
            SimpleRetention(self.hidden_size, gamma, self.head_size, double_v_dim) for gamma in self.gammas
        ])
    def swish(self, x):
        return x * torch.sigmoid(x)
    def forward(self, X):
        """
        parallel representation of the multi-scale retention mechanism
        """

        # apply each individual retention mechanism to X
        Y = []
        for i in range(self.heads):
            Y.append(self.retentions[i](X))

        Y = torch.cat(Y, dim=2)
        Y_shape = Y.shape
        Y = self.group_norm(Y.reshape(-1, self.v_dim)).reshape(Y_shape)

        return (self.swish(X @ self.W_G) * Y) @ self.W_O

    def forward_recurrent(self, x_n, s_n_1s, n):
        """
        recurrent representation of the multi-scale retention mechanism
        x_n: (batch_size, 1, hidden_size)
        s_n_1s: (batch_size, heads, head_size, head_size)

        """

        # apply each individual retention mechanism to a slice of X
        Y = []
        s_ns = []
        for i in range(self.heads):
            y, s_n = self.retentions[i].forward_recurrent(
                x_n[:, :, :], s_n_1s[i], n
            )
            Y.append(y)
            s_ns.append(s_n)

        Y = torch.cat(Y, dim=2)
        Y_shape = Y.shape
        Y = self.group_norm(Y.reshape(-1, self.v_dim)).reshape(Y_shape)

        return (self.swish(x_n @ self.W_G) * Y) @ self.W_O, s_ns

    def forward_chunkwise(self, x_i, r_i_1s, i):
        """
        chunkwise representation of the multi-scale retention mechanism
        x_i: (batch_size, chunk_size, hidden_size)
        r_i_1s: (batch_size, heads, head_size, head_size)
        """
        batch, chunk_size, _ = x_i.shape

        # apply each individual retention mechanism to a slice of X
        Y = []
        r_is = []
        for j in range(self.heads):
            y, r_i = self.retentions[j].forward_chunkwise(
                x_i[:, :, :], r_i_1s[j], i
            )
            Y.append(y)
            r_is.append(r_i)


        Y = torch.cat(Y, dim=2)
        Y_shape = Y.shape
        Y = self.group_norm(Y.reshape(-1, self.v_dim)).reshape(Y_shape)
        x_i = x_i.to(self.W_G.device)
        return (self.swish(x_i @ self.W_G) * Y) @ self.W_O, r_is

class RetNet(nn.Module):
    def __init__(self, layers, hidden_dim, ffn_size, heads, double_v_dim=False):
        super(RetNet, self).__init__()
        self.layers = layers
        self.hidden_dim = hidden_dim
        self.ffn_size = ffn_size
        self.heads = heads
        self.v_dim = hidden_dim * 2 if double_v_dim else hidden_dim

        self.retentions = nn.ModuleList([
            MultiScaleRetention(hidden_dim, heads, double_v_dim)
            for _ in range(layers)
        ])
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, ffn_size),
                nn.GELU(),
                nn.Linear(ffn_size, hidden_dim)
            )
            for _ in range(layers)
        ])
        self.layer_norms_1 = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(layers)
        ])
        self.layer_norms_2 = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(layers)
        ])

    def forward(self, X):
        """
        X: (batch_size, sequence_length, hidden_size)
        """
        for i in range(self.layers):
            Y = self.retentions[i](self.layer_norms_1[i](X)) + X

            X = self.ffns[i](self.layer_norms_2[i](Y)) + Y

        return X

    def forward_recurrent(self, x_n, s_n_1s, n):
        """
        X: (batch_size, sequence_length, hidden_size)
        s_n_1s: list of lists of tensors of shape (batch_size, hidden_size // heads, hidden_size // heads)

        """
        s_ns = []
        for i in range(self.layers):
            # list index out of range
            o_n, s_n = self.retentions[i].forward_recurrent(self.layer_norms_1[i](x_n), s_n_1s[i], n)
            y_n = o_n + x_n
            s_ns.append(s_n)
            x_n = self.ffns[i](self.layer_norms_2[i](y_n)) + y_n

        return x_n, s_ns

    def forward_chunkwise(self, x_i, r_i_1s, i):
        """
        X: (batch_size, sequence_length, hidden_size)
        r_i_1s: list of lists of tensors of shape (batch_size, hidden_size // heads, hidden_size // heads)

        """
        r_is = []
        for j in range(self.layers):
            o_i, r_i = self.retentions[j].forward_chunkwise(self.layer_norms_1[j](x_i), r_i_1s[j], i)
            y_i = o_i + x_i
            r_is.append(r_i)
            x_i = self.ffns[j](self.layer_norms_2[j](y_i)) + y_i

        return x_i, r_is





# %%
import copy
import math
#from utils import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, in_f, out_f):
        '''
        Classify FE feature
        '''
        super(Classifier, self).__init__()
        self.linear_1 = nn.Linear(in_f, out_f)  # 创建一个nn.Linear实例，并将其赋值给self.linear_1，输入的参数为in_f和out_f

    def forward(self, x):
        x = self.linear_1(x)  # 将输入张量x通过self.linear_1处理得到输出张量
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=args.seq_length):

        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        if args.scheme == 'M_O': max_len = max_len // 2 + 1 
        pe = torch.zeros(max_len, d_model) 
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(
            1)  
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (
                    -math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        if args.scheme == 'M_O': pe = torch.cat(
            [pe, pe.flip(dims=(0, 1))[1:]])
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = self.pe[:x.size(0), :]
        return self.dropout(x)


class MSNN_Feature_Embedding_Two_Way(nn.Module):
    def __init__(self):
        super(MSNN_Feature_Embedding_Two_Way, self).__init__()
        # 定义卷积、可分离卷积
        conv = lambda in_f, out_f, kernel, s=None: nn.Sequential(nn.Conv1d(in_f, out_f, (kernel,), stride=s),
                                                                 nn.BatchNorm1d(out_f), nn.LeakyReLU())
        sepconv_same = lambda in_f, out_f, kernel: nn.Sequential(
            nn.Conv1d(in_f, out_f, (kernel,), padding=int(kernel / 2), groups=in_f),
            nn.Conv1d(out_f, out_f, (1,)), nn.BatchNorm1d(out_f), nn.LeakyReLU())

        self.conv_A_0 = conv(1, 4, 5, 1)
        self.sepconv_A_1 = sepconv_same(4, 16, 9)
        self.sepconv_A_2 = sepconv_same(16, 32, 5)
        self.sepconv_A_3 = sepconv_same(32, 64, 3) 

        self.conv_B_0 = conv(1, 4, 10, 1)  
        self.sepconv_B_1 = sepconv_same(4, 16, 9) 
        self.sepconv_B_2 = sepconv_same(16, 32, 5) 
        self.sepconv_B_3 = sepconv_same(32, 64, 3) 
        
        self.conv_C_0 = conv(1, 4, 20, 1)  
        self.sepconv_C_1 = sepconv_same(4, 16, 9) 
        self.sepconv_C_2 = sepconv_same(16, 32, 5) 
        self.sepconv_C_3 = sepconv_same(32, 64, 3) 
        self.ATT = EfficientChannelAttention(112)
        self.gap = nn.AdaptiveAvgPool1d(args.mha_length)  
        self.pe = PositionalEncoding(112,
                                     max_len=args.mha_length)
        self.mha_ff_A = nn.TransformerEncoderLayer(112, args.mha_head,
                                                   dim_feedforward=448)
        self.mha_ff_B = nn.TransformerEncoderLayer(112, args.mha_head,
                                                   dim_feedforward=448) 
        self.mha_ff_C = nn.TransformerEncoderLayer(112, args.mha_head,
                                                   dim_feedforward=448) 

    def seq_trans(self, func, x):
        # 输入形状：[B, F, T]
        return func(x.permute(-1, 0, 1)).permute(1, -1, 0)

    def one_way(self, conv_0, sepconv_1, sepconv_2, sepconv_3, x, mha_ff=None,att = None):
        b, l, t = x.shape
        x = x.reshape(-1, 1, t) #[B*F,1,T]
        x = conv_0(x)
        x = sepconv_1(x)
        x1 = x
        x = sepconv_2(x) 
        x2 = x
        x = sepconv_3(x) 
        x3 = x
        x = self.gap(torch.cat([x1, x2, x3], 1))
        t = self.ATT(x)
        if att !=None:
            x = self.ATT(x+att)
        att = t+x
        x = x + self.seq_trans(self.pe, x) 
        x = x.permute(-1, 0, 1) #[T,B*F,F]
        x = mha_ff(x).permute(1, -1, 0) 
        x = x.reshape(b, l, *x.shape[-2:]) 

        return x , att
    def one_way0(self, conv_0, sepconv_1, sepconv_2, sepconv_3, x, mha_ff=None,att = None):
        #print(x.shape)
        b, l, t = x.shape
        x = x.reshape(-1, 1, t) #[B*F,1,T]
        #print(x.shape)
        x = conv_0(x)
        x = sepconv_1(x)
        x1 = x
        x = sepconv_2(x) 
        x2 = x
        x = sepconv_3(x) 
        x3 = x
        #print(x.shape)
        x = self.gap(torch.cat([x1, x2, x3], 1))
        #print(x.shape)
        x = x + self.seq_trans(self.pe, x) 
        #print(x.shape)
        x = x.permute(-1, 0, 1) #[T,B*F,F]
        #print(x.shape)
        x = mha_ff(x).permute(1, -1, 0)
        #print(x.shape)
        x = x.reshape(b, l, *x.shape[-2:]) 
        #print(x.shape)

        return x

    def forward(self, x):
        x_A = self.one_way0(self.conv_A_0, self.sepconv_A_1, self.sepconv_A_2, self.sepconv_A_3, x,
                           mha_ff=self.mha_ff_A) 
        x_B ,att= self.one_way(self.conv_B_0, self.sepconv_B_1, self.sepconv_B_2, self.sepconv_B_3, x,
                           mha_ff=self.mha_ff_B)
        x_C ,att= self.one_way(self.conv_C_0, self.sepconv_C_1, self.sepconv_C_2, self.sepconv_C_3, x,
                           mha_ff=self.mha_ff_C,att = att)
        x = torch.cat((x_A, x_B,x_C), dim=-2) 
        return x
#############################################################################################################################

class Context_Encoder(nn.Module):
    def __init__(self, f, h):
        super(Context_Encoder, self).__init__()

        self.biLSTM = nn.LSTM(f, h, num_layers=args.lstm_layers, bidirectional=True,batch_first=True)
        self.biLSTM = nn.LSTM(f, h ,num_layers=args.lstm_layers, dropout=0.5, bidirectional=True,batch_first=True)
        self.dropout_1 = nn.Dropout() 
        self.dropout_2 = nn.Dropout() 
        self.MR1 = RetNet(layers=1, hidden_dim=f, ffn_size=h, heads=16, double_v_dim=False)
        
    def forward(self, x):  # [B, L, F]
        
        h, _ = self.biLSTM(x)  
        h = self.dropout_1(h) 
        h = h+x
        h = self.dropout_2(h) 
        return h
    def forward_ret(self, x): 
        h = self.MR1(x)  
        h = self.dropout_1(h) 
        h = h+x
        h = self.dropout_2(h) 
        return h
#############################################################################################################################

class Model(nn.Module):
    def __init__(self, x, num_classes=12):
        super(Model, self).__init__()
        self.FE = MSNN_Feature_Embedding_Two_Way().to(device)
        with torch.no_grad():
            b, l, f, t = self.FE(x).shape 
            feature_size = f * t  
            embedding_size = int(feature_size / 2)

        self.Context_Encoder = Context_Encoder(feature_size, embedding_size).to(device) 
        with torch.no_grad(): 
            x = self.FE(x)
            x = x.flatten(start_dim=2)
            feature_size2 = self.Context_Encoder(x).shape[-1]
        self.project_f = nn.Linear(feature_size,feature_size2) 
        self.dropout = nn.Dropout()
        self.cls = Classifier(feature_size2, num_classes)

    def forward(self, x):
        x = x.to(device)
#         print(x.shape)
        x = self.FE(x)
#         print(x.shape)

        b, l, f, t = x.shape  # 计算输入张量x的形状
        x = x.view(b,l, f*t)
#         print(x.shape)
        h = self.Context_Encoder.forward_ret(x)
#         print(h.shape)
        l_2 = self.cls(h.to(device)) 
        l_2 = l_2.flatten(end_dim=1)
#         print(l_2.shape)

        return l_2

# %%
