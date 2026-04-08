import torch
import torch.nn as nn


class BiRNN_Gait_Estimator(nn.Module):
    def __init__(self, config):
        super(BiRNN_Gait_Estimator, self).__init__()

        # 定义双向 LSTM 层
        # batch_first=True 表示输入数据的形状是 (Batch, Seq, Feature)
        self.lstm = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=True,  # 开启双向网络
            dropout=config.dropout_rate if config.num_layers > 1 else 0
        )

        # 全连接层
        # ⚠️ 注意：因为是双向RNN，它会将正向和反向的特征拼接，所以输入维度需要乘以 2
        self.fc = nn.Linear(config.hidden_size * 2, config.output_size)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)

        # output 包含所有时间步的输出
        # h_n 包含最后一个时间步的隐藏状态
        output, (h_n, c_n) = self.lstm(x)

        # 提取双向 LSTM 最后一层的正向和反向隐藏状态
        # h_n shape: (num_layers * 2, batch_size, hidden_size)
        # h_n[-2, :, :] 是正向的最后一层状态
        # h_n[-1, :, :] 是反向的最后一层状态
        forward_hidden = h_n[-2, :, :]
        backward_hidden = h_n[-1, :, :]

        # 将正向和反向的特征在特征维度（dim=1）拼接起来
        # concat_hidden shape: (batch_size, hidden_size * 2)
        concat_hidden = torch.cat((forward_hidden, backward_hidden), dim=1)

        # 通过全连接层输出最终预测的角度
        predictions = self.fc(concat_hidden)

        return predictions