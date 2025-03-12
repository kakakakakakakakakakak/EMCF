from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
import math
import os
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from mamba_ssm import Mamba
import pandas as pd
import numpy as np
# from math import sqrt, ceil
# from sklearn.preprocessing import StandardScaler
from einops import repeat
# from sklearn.preprocessing import MinMaxScaler
# # 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义数据和路径
factors = ['EBL', 'EBR', 'EBT', 'NBL', 'NBR', 'NBT', 'SBL', 'SBR', 'SBT', 'WBL', 'WBR', 'WBT']
Path = "D:\\Desktop\\KAKA\\MY\\LX\\newTraffic\\data\\"

# 加载训练数据
train_df = pd.read_csv(Path + "WLtrain.csv", index_col='Time_in')
train_dataset = train_df.to_numpy()

# 加载测试数据
test_df = pd.read_csv(Path + "WLtest.csv", index_col='Time_in')
test_dataset = test_df.to_numpy()

# 定义 StandardScaler 对象
# scaler_X = StandardScaler()

# scaler_Y = StandardScaler()

# 创建滑动窗口数据集
def create_channel_independent_dataset(dataset, look_back, look_forward):
    num_features = dataset.shape[-1]
    datax, datax1,datax11,datax2,datax21, datax3,datax31,dataY = [], [],[],[],[],[],[],[]
    for i in range(len(dataset) - look_back - look_forward + 1):
        a = dataset[i:(i + look_back), :30]
        a1 = dataset[i:(i + look_back), 12:18]
        a11 = dataset[i + look_back:i + look_back + look_forward, 12:18]
        a2 = dataset[i:(i + look_back), 18:22]
        a21 = dataset[(i + look_back-1):(i + look_back), 18:22]
        a3 = dataset[i:(i + look_back), 22:30]
        a31 = dataset[i + look_back:i + look_back + look_forward, 22:30]
        b = dataset[i + look_back:i + look_back + look_forward, :12]  # 输出特征为12个特征
        datax.append(a)
        datax1.append(a1)
        datax11.append(a11)
        datax2.append(a2)
        datax21.append(a21)
        datax3.append(a3)
        datax31.append(a31)
        dataY.append(b)
    return np.array(datax),np.array(datax1),np.array(datax11),np.array(datax2),np.array(datax21),np.array(datax3),np.array(datax31), np.array(dataY)

look_back = 12
look_forward = 1

trainx,trainx1,trainx11,trainx2,trainx21,trainx3,trainx31,trainY = create_channel_independent_dataset(train_dataset, look_back, look_forward)
testx,testx1,testx11,testx2,testx21,testx3, testx31,testY = create_channel_independent_dataset(test_dataset, look_back, look_forward)
# trainx_scaled = trainx.reshape(-1, trainx.shape[-1])
# testx_scaled = scaler_X.transform(testx.reshape(-1, testx.shape[-1])).reshape(testx.shape)
# trainx1_scaled = scaler_X.fit_transform(trainx1.reshape(-1, trainx1.shape[-1])).reshape(trainx1.shape)
# testx1_scaled = scaler_X.transform(testx1.reshape(-1, testx1.shape[-1])).reshape(testx1.shape)
# trainx11_scaled = scaler_X.fit_transform(trainx11.reshape(-1, trainx11.shape[-1])).reshape(trainx11.shape)
# testx11_scaled = scaler_X.transform(testx11.reshape(-1, testx11.shape[-1])).reshape(testx11.shape)
# trainx2_scaled = scaler_X.fit_transform(trainx2.reshape(-1, trainx2.shape[-1])).reshape(trainx2.shape)
# testx2_scaled = scaler_X.transform(testx2.reshape(-1, testx2.shape[-1])).reshape(testx2.shape)
# trainx21_scaled = scaler_X.fit_transform(trainx21.reshape(-1, trainx21.shape[-1])).reshape(trainx21.shape)
# testx21_scaled = scaler_X.transform(testx21.reshape(-1, testx21.shape[-1])).reshape(testx21.shape)
# trainx3_scaled = scaler_X.fit_transform(trainx3.reshape(-1, trainx3.shape[-1])).reshape(trainx3.shape)
# testx3_scaled = scaler_X.transform(testx3.reshape(-1, testx3.shape[-1])).reshape(testx3.shape)
# trainx31_scaled = scaler_X.fit_transform(trainx31.reshape(-1, trainx31.shape[-1])).reshape(trainx31.shape)
# testx31_scaled = scaler_X.transform(testx31.reshape(-1, testx31.shape[-1])).reshape(testx31.shape)
# trainY_scaled = scaler_Y.fit_transform(trainY.reshape(-1, trainY.shape[-1])).reshape(trainY.shape)
# testY_scaled = scaler_Y.transform(testY.reshape(-1, testY.shape[-1])).reshape(testY.shape)
# # 将数据转换为 PyTorch 张量
trainx_tensor = torch.tensor(trainx, dtype=torch.float32).to(device)
trainx1_tensor = torch.tensor(trainx1, dtype=torch.float32).to(device)
trainx2_tensor = torch.tensor(trainx2, dtype=torch.float32).to(device)
trainx3_tensor = torch.tensor(trainx3, dtype=torch.float32).to(device)
trainx11_tensor = torch.tensor(trainx11, dtype=torch.float32).to(device)
trainx21_tensor = torch.tensor(trainx21, dtype=torch.float32).to(device)
trainx31_tensor = torch.tensor(trainx31, dtype=torch.float32).to(device)
trainY_tensor = torch.tensor(trainY.squeeze(), dtype=torch.float32).to(device)
testx_tensor = torch.tensor(testx, dtype=torch.float32)
testx1_tensor = torch.tensor(testx1, dtype=torch.float32)
testx2_tensor = torch.tensor(testx2, dtype=torch.float32)
testx3_tensor = torch.tensor(testx3, dtype=torch.float32)
testx11_tensor = torch.tensor(testx11, dtype=torch.float32)
testx21_tensor = torch.tensor(testx21, dtype=torch.float32)
testx31_tensor = torch.tensor(testx31, dtype=torch.float32)
testY_tensor = torch.tensor(testY.squeeze(), dtype=torch.float32)

# 创建 TensorDataset 和 DataLoader
train_dataset = TensorDataset(trainx_tensor,trainx1_tensor,trainx2_tensor,trainx3_tensor, trainx11_tensor,trainx21_tensor,trainx31_tensor,trainY_tensor)
test_dataset = TensorDataset(testx_tensor,testx1_tensor,testx2_tensor,testx3_tensor,testx11_tensor,testx21_tensor,testx31_tensor, testY_tensor)

# 设置 DataLoader
train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4096, shuffle=False)
class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads):

        super(MultiHeadAttention, self).__init__()

        # 验证 d_model 是否可以被 num_heads 整除，以便分割为多个头
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model  # 输入的特征维度
        self.num_heads = num_heads  # 注意力头的数量
        self.d_k = d_model // num_heads  # 每个注意力头的维度大小

        # 定义用于生成查询(Q)、键(K)、值(V)的线性变换
        self.W_q = nn.Linear(d_model, d_model)  # 线性层，将输入映射到查询
        self.W_k = nn.Linear(d_model, d_model)  # 线性层，将输入映射到键
        self.W_v = nn.Linear(d_model, d_model)  # 线性层，将输入映射到值

        # 输出线性层，将多头注意力的结果映射回 d_model 的维度
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        attn_probs = torch.softmax(attn_scores, dim=-1)

        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):

        batch_size, seq_length, d_model = x.size()
        # 拆分成 num_heads 个头，并调整维度顺序
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):

        batch_size, _, seq_length, d_k = x.size()
        # 调整维度顺序并合并 num_heads 维度
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):

        # 将输入通过线性层，映射到查询、键、值
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        # 计算多头注意力输出
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)

        # 合并多个头并通过输出线性层
        output = self.W_o(self.combine_heads(attn_output))
        return output

class MultiScaleXLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads, scale=3, batch_first=True, bidirectional=False, num_mem_tokens=8, dropout=0.1):
        super(MultiScaleXLSTM, self).__init__()
        self.scale = scale
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.batch_first = batch_first

        # Create separate XLSTM instances for each scale
        self.xlstms = nn.ModuleList([
            XLSTM(input_size=input_size,
                  hidden_size=hidden_size,
                  num_layers=num_layers,
                  num_heads2=num_heads2,
                  batch_first=batch_first,
                  bidirectional=bidirectional,
                  num_mem_tokens=num_mem_tokens)
            for _ in range(scale)
        ])
        self.attention_layer = AttentionLayer(d_model1)
    def forward(self, x):
        B, L, D = x.size()
        scale = self.scale
        x_scaled = {}

        # 分割输入为多个尺度
        for i in range(1, scale + 1):
            # 每个尺度对应取后 L//i 的部分
            start_idx = L - (L // i)
            x_scaled[f'x{i}'] = x[:, start_idx:, :]  # [B, L_i, D]
        upsampled_outputs = []
        for i in range(1, scale + 1):
            xi = x_scaled[f'x{i}']
            if xi.shape[1] < L:
                # 使用线性插值上采样到长度 L
                xi = xi.transpose(1, 2)  # [B, D, L_i]
                xi = nn.functional.interpolate(xi, size=L, mode='linear', align_corners=False)  # [B, D, L]
                xi = xi.transpose(1, 2)  # [B, L, D]
            elif xi.shape[1] > L:
                # 如果 L_i > L，则裁剪
                xi = xi[:, :L, :]  # [B, L, D]
            # 如果 L_i == L，无需处理
            upsampled_outputs.append(xi)  # [B, L, D]
        # 存储各个尺度的输出
        outputs = []
        for i in range(scale):
            xi = upsampled_outputs[i - 1]
            xlstm = self.xlstms[i]
            # 逐层通过XLSTM
            xi = xlstm(xi)  # [B, L_i, D]
            outputs.append(xi)  # [B, L_i, D]

        merged_output = torch.stack(outputs, dim=1)  # [B, L, D]
        output = self.attention_layer(merged_output)
        return output
class XLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads2):
        super(XLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.forget_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.input_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.cell_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.output_gate = nn.Linear(input_size + hidden_size, hidden_size)

        # Multihead attention layer
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads2)
        self.attention_layer = nn.Sequential(nn.Linear(hidden_size, hidden_size*2),
                               nn.GELU(),
                               nn.Linear(hidden_size*2, hidden_size))

        self.layer_norm_attn = nn.LayerNorm(hidden_size)
        self.layer_norm_out = nn.LayerNorm(hidden_size)
    def forward(self, x, hx, mem_tokens):
        h_prev, c_prev = hx
        combined = torch.cat([x, h_prev], dim=1)

        f_t = torch.sigmoid(self.forget_gate(combined))
        i_t = torch.sigmoid(self.input_gate(combined))
        c_tilde = torch.tanh(self.cell_gate(combined))
        c_t = f_t * c_prev + i_t * c_tilde

        # Enhanced output gate with multi-head attention mechanism
        o_t = torch.sigmoid(self.output_gate(combined))

        # Prepare the query, key, and value for attention
        h_t = o_t * torch.tanh(c_t)
        h_t = h_t.unsqueeze(0)  # Adding sequence dimension for attention ([1, batch_size, hidden_size])

        # If memory tokens are provided, concatenate them to the current hidden state
        if mem_tokens is not None:
            mem_tokens = repeat(mem_tokens, 'm d -> m b d', b=h_t.size(1))   # Repeat mem_tokens for the batch
            combined_tokens = torch.cat([mem_tokens, h_t], dim=0)  # [mem_len + 1, batch_size, hidden_size]
            attn_output, _ = self.attention(h_t, h_t, h_t)
            attn_output = self.attention_layer(attn_output)
        else:
            attn_output, _ = self.attention(h_t, h_t, h_t)
            attn_output = self.attention_layer(attn_output)

        # Remove the sequence dimension
        attn_output = attn_output.squeeze(0)
        # attn_output = self.layer_norm_attn(attn_output)

        # 残差连接
        h_t = attn_output + h_prev  # 假设残差连接是将注意力输出与前一隐藏状态相加

        # # 进一步的层归一化（可选）
        h_t = self.layer_norm_out(h_t)
        return h_t, c_t
class XLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads2, batch_first=True, bidirectional=False, num_mem_tokens=6):
        super(XLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads2 = num_heads2
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.num_mem_tokens = num_mem_tokens
        self.dropout = nn.Dropout(dropout)
        self.forward_layers = nn.ModuleList([
            XLSTMCell(input_size if i == 0 else hidden_size, hidden_size, num_heads2=num_heads2) for i in
            range(num_layers)
        ])
        if self.bidirectional:
            self.backward_layers = nn.ModuleList([
                XLSTMCell(input_size if i == 0 else hidden_size, hidden_size, num_heads2=num_heads2) for i in
                range(num_layers)
            ])

        # Initialize memory tokens
        if num_mem_tokens > 0:
            self.mem_tokens = nn.Parameter(torch.randn(num_mem_tokens, hidden_size) * 0.01)
        else:
            self.mem_tokens = None

        self.forward_residual_projs = nn.ModuleList([
            nn.Linear(input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)
        ])

        if self.bidirectional:
            self.backward_residual_projs = nn.ModuleList([
                nn.Linear(input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)
            ])
        self.layer_norm = nn.LayerNorm(hidden_size * (2 if bidirectional else 1))
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        if self.batch_first:
            x = x.permute(1, 0, 2)  # [seq_len, batch_size, input_size]

        h = [torch.zeros(batch_size, self.hidden_size).to(x.device) for _ in range(self.num_layers)]
        c = [torch.zeros(batch_size, self.hidden_size).to(x.device) for _ in range(self.num_layers)]

        outputs = []
        for t in range(seq_len):
            input_t = x[t]
            for layer_idx, layer in enumerate(self.forward_layers):
                # 记录输入以用于残差连接
                residual = input_t
                # 前向传播
                h[layer_idx], c[layer_idx] = layer(input_t, (h[layer_idx], c[layer_idx]), mem_tokens=self.mem_tokens)
                residual = self.forward_residual_projs[layer_idx](residual)
                # 残差连接
                input_t = self.dropout(h[layer_idx] + residual)
            outputs.append(input_t)

        if self.bidirectional:
            # Backward pass
            h_back = [torch.zeros(batch_size, self.hidden_size, device=x.device) for _ in range(self.num_layers)]
            c_back = [torch.zeros(batch_size, self.hidden_size, device=x.device) for _ in range(self.num_layers)]
            backward_outputs = []
            for t in reversed(range(seq_len)):
                input_t = x[t]
                for layer_idx, layer in enumerate(self.backward_layers):
                    # 记录输入以用于残差连接
                    residual_back = input_t
                    # 反向传播
                    h_back[layer_idx], c_back[layer_idx] = layer(input_t, (h_back[layer_idx], c_back[layer_idx]), mem_tokens=self.mem_tokens)
                    residual_back = self.backward_residual_projs[layer_idx](residual_back)
                    input_t = self.dropout(h_back[layer_idx] + residual_back)
                backward_outputs.append(input_t)
            backward_outputs.reverse()

            outputs = [torch.cat([f, b], dim=1) for f, b in zip(outputs, backward_outputs)]
            outputs = [self.layer_norm(out) for out in outputs]
        outputs = torch.stack(outputs, dim=0)  # [seq_len, batch_size, hidden_size]
        if self.batch_first:
            outputs = outputs.permute(1, 0, 2)  # [batch_size, seq_len, hidden_size]

        return outputs

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()

        # 初始化一个用于存储位置编码的张量，形状为 (max_seq_length, d_model)
        pe = torch.zeros(max_seq_length, d_model)

        # 创建一个张量，表示序列中的位置，形状为 (max_seq_length, 1)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)

        # 计算位置编码中的除数项（指数衰减），形状为 (d_model // 2)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        # 将正弦编码赋值到偶数维度的位置
        pe[:, 0::2] = torch.sin(position * div_term)

        # 将余弦编码赋值到奇数维度的位置
        pe[:, 1::2] = torch.cos(position * div_term)

        # 扩展一个批次维度（使其形状为 (1, max_seq_length, d_model)），并将其注册为模型的非参数缓冲区
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # 将位置编码与输入相加。取位置编码中与输入序列长度匹配的部分
        return x + self.pe[:, :x.size(1)]
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()

        # 定义前馈网络的第一层，全连接层，将输入映射到高维空间
        self.fc1 = nn.Linear(d_model, d_model*2)

        # 定义前馈网络的第二层，全连接层，将高维特征映射回原始维度
        self.fc2 = nn.Linear(d_model*2, d_model)

        # 激活函数，ReLU 用于引入非线性
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
class EncoderLayer(nn.Module):
    def __init__(self, d_model1,num_heads, d_ff,d_ff1, dropout,d_state,d_model,num_heads2):
        super(EncoderLayer, self).__init__()
        self.mamba_forward_list = nn.ModuleList([
            Mamba(d_model1, d_state=d_state, d_conv=3, expand=1) for _ in range(d_model)
        ])
        self.mamba_backward_list = nn.ModuleList([
            Mamba(d_model1, d_state=d_state, d_conv=3, expand=1) for _ in range(d_model)
        ])

        # 多头自注意力模块
        # self.self_attn = MultiHeadAttention(d_model1, num_heads)

        # 位置前馈网络模块
        # self.feed_forward = PositionWiseFeedForward(d_model1, d_ff)
        self.cross_attn1 = nn.MultiheadAttention(d_model, num_heads)
        self.cross_attn2 = nn.MultiheadAttention(d_model1, num_heads2)
        # 第一层归一化，用于多头自注意力后的残差连接
        self.norm1 = nn.LayerNorm(d_model)
        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_model*2),
                                  nn.GELU(),
                                  nn.Linear(d_model*2, d_model))
        self.MLP2 = nn.Sequential(nn.Linear(d_model1, d_ff1),
                                  nn.GELU(),
                                  nn.Linear(d_ff1, d_model1))
        # 第二层归一化，用于前馈网络后的残差连接
        self.norm2 = nn.LayerNorm(d_model1)
        self.dense = nn.Linear(18,d_model1)
        # Dropout 层，用于防止过拟合
        self.dropout = nn.Dropout(dropout)
        self.channel_weights = nn.Parameter(torch.ones(d_model), requires_grad=True)
    def forward(self, x,x123):
        all_outputs = []
        # x = x.permute(0,2,1)
        for i in range(x.size(1)):  # 遍历每个通道
            channel_x = x[:, i, :].unsqueeze(1)

            forward_out = self.mamba_forward_list[i](channel_x)
            # backward_out = self.mamba_backward_list[i](torch.flip(channel_x, dims=[2]))
            # backward_out = torch.flip(backward_out, dims=[2])
            # 合并前向和后向的输出
            # combined_out1 = forward_out + backward_out

            weighted_out = self.channel_weights[i] * forward_out
            all_outputs.append(weighted_out)
        all_outputs = torch.stack(all_outputs, dim=1)  #
        all_outputs = all_outputs.view(all_outputs.size(0), -1, all_outputs.size(-1)) #(B,D,L)
        # all_outputs = self.selfdense6(all_outputs)
        all_outputs = (all_outputs + x).permute(2,0,1)
        x123 = self.dense(x123)
        # all_outputs = x.permute(2, 0, 1)
        attn_output,_ = self.cross_attn1(all_outputs, x.permute(2,0,1), x.permute(2,0,1))
        attn_output = self.norm1(all_outputs + self.dropout(attn_output))
        all_outputs = attn_output + self.dropout(self.MLP1(attn_output))
        x = (self.norm1(all_outputs)).permute(2,1,0)
        # x = attn_output.permute(2,1,0)
        x_att,_ = self.cross_attn2(x,x123.permute(1,0,2),x123.permute(1,0,2))
        # x_att, _ = self.cross_attn2(x, x_att, x_att)
        x_att = self.norm2(x + self.dropout(x_att))
        x_att = x_att + self.dropout(self.MLP2(x_att))
        x = (self.norm2(x_att)).permute(1,0,2)
        # x = x_att.permute(1, 2, 0)
        # x = all_outputs.permute(1,0,2)
        return x
class Encoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, scale=4):
        super(Encoder, self).__init__()
        self.scale = scale
        self.num_layers = num_layers
        self.encoder_layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        # self.norm = nn.LayerNorm(encoder_layer.d_model1)

        self.attention_layer = AttentionLayer(d_model1)
    def forward(self, x, x123):

        B, L, D = x.shape
        scale = self.scale
        x_scaled = {}

        # 分割输入为多个尺度
        for i in range(1, scale + 1):
            # 每个尺度对应取后 L//i 的部分
            start_idx = L - (L // i)
            x_scaled[f'x{i}'] = x[:, start_idx:, :]  # [B, L_i, D]
        upsampled_outputs = []
        for i in range(1, scale + 1):
            xi = x_scaled[f'x{i}'] # [B, L_i, D]
            if xi.shape[1] < L:
                # 使用线性插值上采样到长度 L
                xi = xi.transpose(1, 2)  # [B, D, L_i]
                xi = nn.functional.interpolate(xi, size=L, mode='linear', align_corners=False)  # [B, D, L]
                xi = xi.transpose(1, 2)  # [B, L, D]
            elif xi.shape[1] > L:
                # 如果 L_i > L，则裁剪
                xi = xi[:, :L, :]  # [B, L, D]
            # 如果 L_i == L，无需处理
            upsampled_outputs.append(xi)  # [B, L, D]
        # 存储各个尺度的输出
        outputs = []
        # accumulated_xi = torch.zeros_like(upsampled_outputs[0])
        for i in range(1, scale + 1):
            xi = upsampled_outputs[i - 1]  # [B, L_i, D]
            # 将之前累积的 xi 加到当前 xi 上
            # xi = xi + accumulated_xi
            # 逐层通过编码器
            for layer in self.encoder_layers:
                xi = layer(xi, x123)  # [B, L_i, D]
            outputs.append(xi)  # [B, L_i, D]
            # 更新累积的 xi
            # accumulated_xi = accumulated_xi + xi

        merged_output = torch.stack(outputs, dim=1)  # [B, L, D]
        output = self.attention_layer(merged_output)
        return output
class AttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super(AttentionLayer, self).__init__()
        self.attn_weights = nn.Linear(input_dim, 1, bias=False)

    def forward(self, x):

        scores = self.attn_weights(x)
        weights = torch.softmax(scores, dim=1)
        weighted_sum = torch.sum(x * weights, dim=1)
        return weighted_sum

class BidirectionalMambaModel(nn.Module):
    def __init__(self, d_model, d_state, d_ff,d_ff1, output_dim, dropout, num_heads, d_model1, num_heads2,d_model2):
        super(BidirectionalMambaModel, self).__init__()

        self.layernorm = nn.LayerNorm(6)
        self.bn = nn.BatchNorm1d(6)
        self.bn1 = nn.BatchNorm1d(12)
        self.bn2 = nn.BatchNorm1d(48)
        self.bn3 = nn.BatchNorm1d(60)
        self.bn4 = nn.BatchNorm1d(36)
        self.layernorm1 = nn.LayerNorm(12)
        self.layernorm2 = nn.LayerNorm(48)
        self.layernorm3 = nn.LayerNorm(60)
        self.layernorm4 = nn.LayerNorm(36)
        self.layernorm5 = nn.LayerNorm(60)

        # 双向Mamba层
        self.dense1 = nn.Linear(18,d_model)
        self.dense2 = nn.Linear(18, 36)
        self.dense3 = nn.Linear(12, 24)
        # self.dense4 = nn.Linear(60, 48)
        self.dense5 = nn.Linear(d_model1, d_ff1)
        self.dense10 = nn.Linear(d_model1, d_ff1)
        self.dense33 = nn.Linear(d_model*3 , d_model*3 )
        self.linear = nn.Linear(d_ff1,d_model2)
        # self.dense5 = nn.Linear(60, 48)
        self.selfdense12 = nn.Linear(18,12)
        self.selfdense6 = nn.Linear(36, d_model1)
        self.selfdense321 = nn.Linear(18,  6)

        self.cross_attention1 = nn.MultiheadAttention(embed_dim=d_ff, num_heads=num_heads, dropout=dropout)
        # self.cross_attention2 = nn.MultiheadAttention(embed_dim=d_model2, num_heads=num_heads2, dropout=dropout)
        # self.dropout = nn.Dropout(0.2)
        # 时间相关性编码层 - 使用双向LSTM替代卷积操作
        self.x_lstm1 = MultiScaleXLSTM(
            input_size=d_model1,
            hidden_size=d_model1,
            num_layers=1,
            num_heads=num_heads,
            scale=2,
            batch_first=True,
            bidirectional=False,
            num_mem_tokens=num_mem_tokens,
            dropout=dropout
        )
        # self.x_lstm_layernorm1 = nn.LayerNorm(d_ff1)
        self.dropout = nn.Dropout(0.2)

        self.projector = nn.Linear(d_ff1, output_dim, bias=True)  # 将 D 映射到 120

        self.attention_layer = AttentionLayer(d_ff1)

        self.encoder_embedding = nn.Linear(30, d_model1)

        # 位置编码
        self.positional_encoding = PositionalEncoding(d_model1, max_seq_length)
        # 编码器层
        encoder_layer = EncoderLayer(
            d_model1=d_model1,
            num_heads=num_heads,
            d_ff=d_ff,
            d_ff1=d_ff1,
            dropout=dropout,
            d_state=d_state,
            d_model=d_model,
            num_heads2=num_heads2
        )
        self.encoder = Encoder(
            encoder_layer=encoder_layer,
            num_layers=2,
            scale=3
        )
        self.MLP1 = nn.Sequential(nn.Linear(d_ff1, d_ff1*2),
                              nn.GELU(),
                              nn.Linear(d_ff1*2, d_ff1))
        self.MLP2 = nn.Sequential(nn.Linear(d_model2, d_model1),
                              nn.GELU(),
                              nn.Linear(d_model1 , d_model2))
        self.norm1  = nn.LayerNorm(d_ff1)
        self.norm2 = nn.LayerNorm(d_model2)

        # self.scale= 4
    # def generate_mask(self, x):
    #   x_mask = (x.sum(dim=2) != 0).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, src_len)

    def forward(self, x,x1,x2,x3,x11,x21,x31):
        # x1 = nn.ReLU()(x1)
        # x2 = nn.ReLU()(x2)
        # x3 = nn.ReLU()(x3)
        x123 = torch.cat((x1,x2,x3),dim=2)
        # x123 = self.selfdense12(x123)
        # x123 = self.layernorm1(x123)
        # x123 = nn.ReLU()(x123)
        # x321 = torch.cat((x11, x21,x31), dim=2)
        # x321 = self.selfdense321(x321)
        # x321 = nn.ReLU()(x321)
        # x321 = self.layernorm(x321)
        # x321 = torch.cat((x321, x321, x321,x321,x321,x321,x321,x321,x321,x321,x321,x321), dim=1)
        # x = x.permute(0, 2, 1)
        # x = torch.cat((x, x321), dim=2)
        # x= self.dense1(x)
        # x = self.dense3(x.permute(0,2,1))
        # # x = nn.ReLU()(x)
        # x= torch.cat((x,x123),dim=2)

        # x_mask = self.generate_mask(x)
        # enc_out, n_vars = self.patch_embedding(x)
        x_embedded = self.encoder_embedding(x)
        x_embedded = x_embedded + self.dropout(self.positional_encoding(x_embedded))
        # enc_output = x_embedded
        enc_output = self.encoder(x_embedded, x123)  # [B, L, D]

        # x = self.selfdense6(x)

        all_outputs =  enc_output + x_embedded

        res = self.dense5(all_outputs)


        # 时间相关性编码层 - 使用双向LSTM
        lstm_out = self.x_lstm1(all_outputs)
        # lstm_out = self.x_lstm_layernorm1(lstm_out)
        lstm_out = self.dropout(lstm_out)
        lstm_out = self.dense10(lstm_out)

        lstm_out =lstm_out  + res
        #
        # lstm_out = lstm_out.permute(2, 0, 1)
        # attn_output2, _ = self.cross_attention1(lstm_out, lstm_out,  lstm_out)
        # all_outputs = self.norm1(lstm_out + self.dropout(attn_output2))
        all_outputs = lstm_out + self.dropout(self.MLP1(lstm_out))
        attn_output2 = (all_outputs)
        # #
        # all_outputs6 = all_outputs.permute(1, 2, 0)
        pooled_output = self.attention_layer(attn_output2)
        output = self.projector(pooled_output)
        return output
    # 模型参数定义


# EXD_dim = 18
enc_in = 30
input_dim = 12
output_dim = 12
d_model = 12
d_model1 = 64
d_model2 = 32
d_state = 6
lstm_n = 12
x_lstm_n = 12
d_ff = 12
d_ff1 = 128
dropout = 0.2
num_heads = 4
num_layers = 2
max_seq_length = 12
num_heads2 = 8
num_mem_tokens=6
# patch_len=6
# stride= 6
# 实例化模型，并移动到GPU
model = BidirectionalMambaModel( d_model, d_state, d_ff,d_ff1, output_dim, dropout, num_heads,d_model1,num_heads2,d_model2).to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
#
# # 训练模型
num_epochs = 500
patience = 20  # 早期停止参数
best_val_loss = np.inf
early_stop_counter = 0

#checkpoint_path = r"D:\Desktop\KAKA\MY\LX\S-D-Mamba-main\Abalation\best_checkpoint_path\no_VFL_20250310_162936.pth"
checkpoint_dir = 'best_checkpoint_path'

# 获取当前时间并格式化为 YYYYMMDD_HHMMSS
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
# 获取当前脚本的名称
script_name = os.path.splitext(os.path.basename(__file__))[0]  # 获取代码的文件名，不包含扩展名
# 创建保存文件夹（如果不存在的话）
os.makedirs(checkpoint_dir, exist_ok=True)
# 生成新的保存文件名：'脚本名称_当前时间.pth'
checkpoint_filename = f"{script_name}_{current_time}.pth"
# 构造完整的保存路径
checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
# 假设你有一个模型变量，比如 `model`，将其保存
torch.save(model.state_dict(), checkpoint_path)
history = {'loss': [], 'val_loss': []}
time_start2 = time.time()
#
for epoch in range(num_epochs):
    time_start1 = time.time()
    model.train()
    running_loss = 0.0
    for x,x1,x2,x3,x11,x21,x31, labels in train_loader:
        x,x1,x2,x3,x11,x21,x31, labels = x.to(device),x1.to(device),x2.to(device),x3.to(device),x11.to(device),x21.to(device),x31.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(x,x1,x2,x3,x11,x21,x31)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    history['loss'].append(avg_loss)

    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for x,x1,x2,x3,x11,x21,x31,  labels in test_loader:
            x,x1,x2,x3,x11,x21,x31,  labels = x.to(device),x1.to(device),x2.to(device),x3.to(device),x11.to(device),x21.to(device),x31.to(device), labels.to(device)
            outputs = model(x,x1,x2,x3,x11,x21,x31)
            val_loss = criterion(outputs, labels)
            total_val_loss += val_loss.item()
    avg_val_loss = total_val_loss / len(test_loader)
    history['val_loss'].append(avg_val_loss)
    time_end1 = time.time()

    # 打印训练信息
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    print('once cost', time_end1 - time_start1, 's')
    # 检查是否有改进
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        early_stop_counter = 0
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved better model to {checkpoint_path}")
    else:
        early_stop_counter += 1
        print(f"No improvement in validation loss for {early_stop_counter} epochs.")
        if early_stop_counter >= patience:
            print(f"Stopping early after {patience} epochs without improvement.")
            break

time_end2 = time.time()
print('Training complete. Total time cost:', time_end2 - time_start2, 's')


# 加载最佳模型
model.load_state_dict(torch.load(checkpoint_path))
model.eval()

predictions = []
actual_values = []

# 逐批进行预测并收集结果
with torch.no_grad():
    for batch_x, batch_x1,batch_x2,batch_x3,batch_x11,batch_x21,batch_x31,batch_y in test_loader:
        # 将输入数据放入 GPU 进行预测
        batch_predictions = model(batch_x.to(device),batch_x1.to(device),batch_x2.to(device),batch_x3.to(device),batch_x11.to(device),batch_x21.to(device),batch_x31.to(device)).cpu().numpy()

        # 收集预测值和实际值
        predictions.append(batch_predictions)
        actual_values.append(batch_y.numpy())

# 将所有批量的预测和实际值拼接成完整的数组

# 将列表转换为 NumPy 数组
predictions = np.concatenate(predictions, axis=0)
actual_values = np.concatenate(actual_values, axis=0)

# 确保 predictions 和 actual_values 是二维数组
predictions = np.array(predictions).reshape(-1, predictions.shape[-1])  # 使其为二维数组
actual_values = np.array(actual_values).reshape(-1, actual_values.shape[-1])  # 使其为二维数组

# 反归一化预测值和实际值
# predictions_unnormalized = scaler_Y.inverse_transform(predictions)
# actual_values_unnormalized = scaler_Y.inverse_transform(actual_values)
# predictions = np.concatenate(predictions, axis=0)
# actual_values = np.concatenate(actual_values, axis=0)

# 计算评估指标
evaluation_metrics = {}
for i, factor in enumerate(factors):
    actual = actual_values[:, i]  # 反归一化后的真实标签值
    pred = predictions[:, i]  # 反归一化后的预测值
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)

    evaluation_metrics[factor] = {'RMSE': rmse, 'MAE': mae, 'R2': r2}

# 打印评估结果
for factor, metrics in evaluation_metrics.items():
    print(f"Factor: {factor}, RMSE: {metrics['RMSE']:.4f}, MAE: {metrics['MAE']:.4f}, R²: {metrics['R2']:.4f}")

# 计算平均评估指标
average_rmse = np.mean([metrics['RMSE'] for metrics in evaluation_metrics.values()])
average_mae = np.mean([metrics['MAE'] for metrics in evaluation_metrics.values()])
average_r2 = np.mean([metrics['R2'] for metrics in evaluation_metrics.values()])

print(f"Average RMSE: {average_rmse:.4f}, Average MAE: {average_mae:.4f}, Average R²: {average_r2:.4f}")


#
# test_time_index = pd.to_datetime(test_df.index).to_numpy()
# num_points = min(96, len(test_time_index))  # 避免超出数据范围
# num_cols = 3  # 每行子图的数量
# num_rows = (len(factors) + num_cols - 1) // num_cols  # 计算行数
#
# fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(8*num_cols, 6*num_rows), squeeze=False)
#
# for i, factor in enumerate(factors):
#     actual_values_subset = actual_values[1000:1000+num_points, i]  # 只取前2000个数据点的实际值
#     predicted_values_subset = predictions[1000:1000+num_points, i]  # 只取前2000个数据点的预测值
#     time_subset = test_time_index[:num_points]  # 对应的时间索引
#
#     row_index = i // num_cols
#     col_index = i % num_cols
#
#     ax = axs[row_index, col_index]
#     # 为实际值选择蓝色，线条较粗
#     ax.plot(time_subset, actual_values_subset, label='Actual Value', color='#9EC1D4', linewidth=3)
#     # 为预测值选择红色，线条较细
#     ax.plot(time_subset, predicted_values_subset, label='Prediction', color='#E87651', linewidth=3)
#     ax.set_title(f"Factor: {factor}\nRMSE: {evaluation_metrics[factor]['RMSE']:.2f}, MAE: {evaluation_metrics[factor]['MAE']:.2f}")
#     ax.set_xlabel('Time')
#     ax.set_ylabel('Flow Volume')
#     ax.legend()
#
# # 隐藏多余的子图
# for i in range(len(factors), num_rows * num_cols):
#     axs.flatten()[i].axis('off')
#
# plt.tight_layout()
# plt.show()
# #