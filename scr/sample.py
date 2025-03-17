from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
import math
import os
from datetime import datetime
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from mamba_ssm import Mamba
import pandas as pd
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
factors = ['EBL', 'EBR', 'EBT', 'NBL', 'NBR', 'NBT', 'SBL', 'SBR', 'SBT', 'WBL', 'WBR', 'WBT']
Path = "D:data\\"
train_df = pd.read_csv(Path + "sample_train.csv", index_col='Time_in')
train_dataset = train_df.to_numpy()
test_df = pd.read_csv(Path + "sample_test.csv", index_col='Time_in')
test_dataset = test_df.to_numpy()

def create_channel_independent_dataset(dataset, look_back, look_forward):
    datax, datax1,datax11,datax2,datax21, datax3,datax31,dataY = [], [],[],[],[],[],[],[]
    for i in range(len(dataset) - look_back - look_forward + 1):
        a = dataset[i:(i + look_back), :12]
        a1 = dataset[i:(i + look_back), 12:18]
        a11 = dataset[i + look_back:i + look_back + look_forward, 12:18]
        a2 = dataset[i:(i + look_back), 18:22]
        a21 = dataset[(i + look_back-1):(i + look_back), 18:22]
        a3 = dataset[i:(i + look_back), 22:30]
        a31 = dataset[i + look_back:i + look_back + look_forward, 22:30]
        b = dataset[i + look_back:i + look_back + look_forward, :12]
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

train_dataset = TensorDataset(trainx_tensor,trainx1_tensor,trainx2_tensor,trainx3_tensor, trainx11_tensor,trainx21_tensor,trainx31_tensor,trainY_tensor)
test_dataset = TensorDataset(testx_tensor,testx1_tensor,testx2_tensor,testx3_tensor,testx11_tensor,testx21_tensor,testx31_tensor, testY_tensor)

train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4096, shuffle=False)
class MultiScaleXLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads, scale=3, batch_first=True, bidirectional=False, num_mem_tokens=8, dropout=0.1):
        super(MultiScaleXLSTM, self).__init__()
        self.scale = scale
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.batch_first = batch_first

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
        for i in range(1, scale + 1):
            start_idx = L - (L // i)
            x_scaled[f'x{i}'] = x[:, start_idx:, :]
        upsampled_outputs = []
        for i in range(1, scale + 1):
            xi = x_scaled[f'x{i}']
            if xi.shape[1] < L:
                xi = xi.transpose(1, 2)
                xi = nn.functional.interpolate(xi, size=L, mode='linear', align_corners=False)
                xi = xi.transpose(1, 2)
            elif xi.shape[1] > L:
                xi = xi[:, :L, :]
            upsampled_outputs.append(xi)
        outputs = []
        for i in range(scale):
            xi = upsampled_outputs[i - 1]
            xlstm = self.xlstms[i]
            xi = xlstm(xi)
            outputs.append(xi)
        merged_output = torch.stack(outputs, dim=1)
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
        o_t = torch.sigmoid(self.output_gate(combined))
        h_t = o_t * torch.tanh(c_t)
        h_t = h_t.unsqueeze(0)
        if mem_tokens is not None:
            # mem_tokens = repeat(mem_tokens, 'm d -> m b d', b=h_t.size(1))
            # combined_tokens = torch.cat([mem_tokens, h_t], dim=0)
            attn_output, _ = self.attention(h_t, h_t, h_t)
            attn_output = self.attention_layer(attn_output)
        else:
            attn_output, _ = self.attention(h_t, h_t, h_t)
            attn_output = self.attention_layer(attn_output)

        attn_output = attn_output.squeeze(0)
        attn_output = self.layer_norm_attn(attn_output)

        h_t = attn_output + h_prev

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
            x = x.permute(1, 0, 2)

        h = [torch.zeros(batch_size, self.hidden_size).to(x.device) for _ in range(self.num_layers)]
        c = [torch.zeros(batch_size, self.hidden_size).to(x.device) for _ in range(self.num_layers)]

        outputs = []
        for t in range(seq_len):
            input_t = x[t]
            for layer_idx, layer in enumerate(self.forward_layers):
                residual = input_t
                h[layer_idx], c[layer_idx] = layer(input_t, (h[layer_idx], c[layer_idx]), mem_tokens=self.mem_tokens)
                residual = self.forward_residual_projs[layer_idx](residual)
                input_t = self.dropout(h[layer_idx] + residual)
            outputs.append(input_t)
        if self.bidirectional:
            h_back = [torch.zeros(batch_size, self.hidden_size, device=x.device) for _ in range(self.num_layers)]
            c_back = [torch.zeros(batch_size, self.hidden_size, device=x.device) for _ in range(self.num_layers)]
            backward_outputs = []
            for t in reversed(range(seq_len)):
                input_t = x[t]
                for layer_idx, layer in enumerate(self.backward_layers):
                    residual_back = input_t
                    h_back[layer_idx], c_back[layer_idx] = layer(input_t, (h_back[layer_idx], c_back[layer_idx]), mem_tokens=self.mem_tokens)
                    residual_back = self.backward_residual_projs[layer_idx](residual_back)
                    input_t = self.dropout(h_back[layer_idx] + residual_back)
                backward_outputs.append(input_t)
            backward_outputs.reverse()
            outputs = [torch.cat([f, b], dim=1) for f, b in zip(outputs, backward_outputs)]
            outputs = [self.layer_norm(out) for out in outputs]
        outputs = torch.stack(outputs, dim=0)
        if self.batch_first:
            outputs = outputs.permute(1, 0, 2)
        return outputs
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
class EncoderLayer(nn.Module):
    def __init__(self, d_model1,num_heads, d_ff,d_ff1, dropout,d_state,d_model,num_heads2):
        super(EncoderLayer, self).__init__()
        self.mamba_forward_list = nn.ModuleList([
            Mamba(d_model1, d_state=d_state, d_conv=2, expand=1) for _ in range(d_model)
        ])
        self.cross_attn1 = nn.MultiheadAttention(d_model, num_heads)
        self.cross_attn2 = nn.MultiheadAttention(d_model1, num_heads2)
        self.norm1 = nn.LayerNorm(d_model)
        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_model*2),
                                  nn.GELU(),
                                  nn.Linear(d_model*2, d_model))
        self.MLP2 = nn.Sequential(nn.Linear(d_model1, d_ff1),
                                  nn.GELU(),
                                  nn.Linear(d_ff1, d_model1))
        self.norm2 = nn.LayerNorm(d_model1)
        self.dense = nn.Linear(8,d_model1)
        self.dropout = nn.Dropout(dropout)
        self.channel_weights = nn.Parameter(torch.ones(d_model), requires_grad=True)
    def forward(self, x,x123):
        all_outputs = []
        for i in range(x.size(1)):
            channel_x = x[:, i, :].unsqueeze(1)
            forward_out = self.mamba_forward_list[i](channel_x)
            weighted_out = self.channel_weights[i] * forward_out
            all_outputs.append(weighted_out)
        all_outputs = torch.stack(all_outputs, dim=1)  #
        all_outputs = all_outputs.view(all_outputs.size(0), -1, all_outputs.size(-1))
        all_outputs = (all_outputs + x).permute(2,0,1)
        x123 = self.dense(x123)
        attn_output,_ = self.cross_attn1(all_outputs, all_outputs,all_outputs)
        attn_output = self.norm1(all_outputs + self.dropout(attn_output))
        all_outputs = attn_output + self.dropout(self.MLP1(attn_output))
        x = (self.norm1(all_outputs)).permute(2,1,0)
        x_att,_ = self.cross_attn2(x,x,x)
        x_att = self.norm2(x + self.dropout(x_att))
        x_att = x_att + self.dropout(self.MLP2(x_att))
        x = (self.norm2(x_att)).permute(1,0,2)
        return x
class Encoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, scale=4):
        super(Encoder, self).__init__()
        self.scale = scale
        self.num_layers = num_layers
        self.encoder_layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.attention_layer = AttentionLayer(d_model1)
    def forward(self, x, x123):
        B, L, D = x.shape
        scale = self.scale
        x_scaled = {}
        for i in range(1, scale + 1):
            start_idx = L - (L // i)
            x_scaled[f'x{i}'] = x[:, start_idx:, :]
        upsampled_outputs = []
        for i in range(1, scale + 1):
            xi = x_scaled[f'x{i}']
            if xi.shape[1] < L:
                xi = xi.transpose(1, 2)
                xi = nn.functional.interpolate(xi, size=L, mode='linear', align_corners=False)
                xi = xi.transpose(1, 2)
            elif xi.shape[1] > L:
                xi = xi[:, :L, :]
            upsampled_outputs.append(xi)
        outputs = []
        for i in range(1, scale + 1):
            xi = upsampled_outputs[i - 1]
            for layer in self.encoder_layers:
                xi = layer(xi, x123)
            outputs.append(xi)
        merged_output = torch.stack(outputs, dim=1)
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
        self.layernorm = nn.LayerNorm(4)
        self.layernorm1 = nn.LayerNorm(8)
        self.dense1 = nn.Linear(16,d_model)
        self.dense3 = nn.Linear(12, 24)
        self.dense5 = nn.Linear(d_model1, d_ff1)
        self.dense10 = nn.Linear(d_model1, d_ff1)
        self.selfdense12 = nn.Linear(18,8)
        self.selfdense321 = nn.Linear(18,  4)
        self.cross_attention1 = nn.MultiheadAttention(embed_dim=d_ff, num_heads=num_heads, dropout=dropout)
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
        self.dropout = nn.Dropout(0.2)
        self.projector = nn.Linear(d_ff1, output_dim, bias=True)
        self.attention_layer = AttentionLayer(d_ff1)
        self.encoder_embedding = nn.Linear(32, d_model1)
        self.positional_encoding = PositionalEncoding(d_model1, max_seq_length)
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
        self.norm1  = nn.LayerNorm(d_ff1)

    def forward(self, x,x1,x2,x3,x11,x21,x31):
        x123 = torch.cat((x1,x2,x3),dim=2)
        x123 = self.selfdense12(x123)
        x123 = self.layernorm1(x123)
        x123 = nn.ReLU()(x123)
        x321 = torch.cat((x11, x21,x31), dim=2)
        x321 = self.selfdense321(x321)
        x321 = nn.ReLU()(x321)
        x321 = self.layernorm(x321)
        x321 = torch.cat((x321, x321, x321,x321,x321,x321,x321,x321,x321,x321,x321,x321), dim=1)
        x = x.permute(0, 2, 1)
        x = torch.cat((x, x321), dim=2)
        x= self.dense1(x)
        x = self.dense3(x.permute(0,2,1))
        x= torch.cat((x,x123),dim=2)
        x_embedded = self.encoder_embedding(x)
        x_embedded = x_embedded + self.dropout(self.positional_encoding(x_embedded))
        enc_output = self.encoder(x_embedded, x123)
        all_outputs =  enc_output + x_embedded
        res = self.dense5(all_outputs)
        lstm_out = self.x_lstm1(all_outputs)
        lstm_out = self.dropout(lstm_out)
        lstm_out = self.dense10(lstm_out)
        lstm_out =lstm_out  + res
        all_outputs = lstm_out + self.dropout(self.MLP1(lstm_out))
        attn_output2 = self.norm1(all_outputs)
        pooled_output = self.attention_layer(attn_output2)
        output = self.projector(pooled_output)
        return output

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

model = BidirectionalMambaModel( d_model, d_state, d_ff,d_ff1, output_dim, dropout, num_heads,d_model1,num_heads2,d_model2).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
num_epochs = 500
patience = 20
best_val_loss = np.inf
early_stop_counter = 0
checkpoint_dir = 'best_checkpoint_path'
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
script_name = os.path.splitext(os.path.basename(__file__))[0]
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_filename = f"{script_name}_{current_time}.pth"
checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
torch.save(model.state_dict(), checkpoint_path)
history = {'loss': [], 'val_loss': []}
time_start2 = time.time()

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
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    print('once cost', time_end1 - time_start1, 's')
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
model.load_state_dict(torch.load(checkpoint_path))
model.eval()
predictions = []
actual_values = []
with torch.no_grad():
    for batch_x, batch_x1,batch_x2,batch_x3,batch_x11,batch_x21,batch_x31,batch_y in test_loader:
        batch_predictions = model(batch_x.to(device),batch_x1.to(device),batch_x2.to(device),batch_x3.to(device),batch_x11.to(device),batch_x21.to(device),batch_x31.to(device)).cpu().numpy()
        predictions.append(batch_predictions)
        actual_values.append(batch_y.numpy())
predictions = np.concatenate(predictions, axis=0)
actual_values = np.concatenate(actual_values, axis=0)
predictions = np.array(predictions).reshape(-1, predictions.shape[-1])
actual_values = np.array(actual_values).reshape(-1, actual_values.shape[-1])
evaluation_metrics = {}
for i, factor in enumerate(factors):
    actual = actual_values[:, i]
    pred = predictions[:, i]
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    evaluation_metrics[factor] = {'RMSE': rmse, 'MAE': mae, 'R2': r2}
for factor, metrics in evaluation_metrics.items():
    print(f"Factor: {factor}, RMSE: {metrics['RMSE']:.4f}, MAE: {metrics['MAE']:.4f}, R²: {metrics['R2']:.4f}")
average_rmse = np.mean([metrics['RMSE'] for metrics in evaluation_metrics.values()])
average_mae = np.mean([metrics['MAE'] for metrics in evaluation_metrics.values()])
average_r2 = np.mean([metrics['R2'] for metrics in evaluation_metrics.values()])
print(f"Average RMSE: {average_rmse:.4f}, Average MAE: {average_mae:.4f}, Average R²: {average_r2:.4f}")
