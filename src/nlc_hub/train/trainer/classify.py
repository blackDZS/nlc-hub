from torch import nn
from typing import Optional, List

# MLP 分类头
class MLPClassificationHead(nn.Module):
    def __init__(self, hidden_size, num_labels, mlp_layers: Optional[List[int]] = None, activation=nn.ReLU):
        """
        多层感知机（MLP）分类头。
        适用于输入为固定长度特征向量的分类任务。
        
        参数：
            hidden_size (int): 输入特征维度。
            num_labels (int): 分类类别数。
            mlp_layers (List[int], optional): MLP隐藏层结构，默认为[hidden_size]。
            activation (nn.Module, optional): 激活函数，默认为ReLU。
        输入输出shape示例：
            输入: features.shape = [batch, hidden_size]
            输出: logits.shape = [batch, num_labels]
        """
        super().__init__()
        if mlp_layers is None:
            mlp_layers = [hidden_size]
        layers = []
        in_dim = hidden_size
        for out_dim in mlp_layers:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(activation())
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, num_labels))
        self.classifier = nn.Sequential(*layers)
    def forward(self, features):
        """
        前向传播。
        参数：
            features (Tensor): 输入特征，形状为[batch, hidden_size]
        返回：
            logits (Tensor): 分类logits，形状为[batch, num_labels]
        """
        return self.classifier(features)

# RNN 分类头
class RNNClassificationHead(nn.Module):
    def __init__(self, input_size, hidden_size, num_labels, num_layers=1, dropout=0.1):
        """
        RNN分类头。
        适用于输入为序列特征的分类任务。
        
        参数：
            input_size (int): 输入特征维度。
            hidden_size (int): RNN隐藏层维度。
            num_labels (int): 分类类别数。
            num_layers (int, optional): RNN层数，默认为1。
            dropout (float, optional): dropout概率，默认为0.1。
        输入输出shape示例：
            输入: features.shape = [batch, seq_len, input_size]
            输出: logits.shape = [batch, num_labels]
        """
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
    def forward(self, features):
        """
        前向传播。
        参数：
            features (Tensor): 输入特征，形状为[batch, seq_len, input_size]
        返回：
            logits (Tensor): 分类logits，形状为[batch, num_labels]
        """
        # features: [batch, seq_len, input_size]
        output, _ = self.rnn(features)
        last_hidden = output[:, -1, :]  # [batch, hidden_size]
        return self.classifier(last_hidden)

# LSTM 分类头
class LSTMClassificationHead(nn.Module):
    def __init__(self, input_size, hidden_size, num_labels, num_layers=1, dropout=0.1):
        """
        LSTM分类头。
        适用于输入为序列特征的分类任务，能捕捉长期依赖。
        
        参数：
            input_size (int): 输入特征维度。
            hidden_size (int): LSTM隐藏层维度。
            num_labels (int): 分类类别数。
            num_layers (int, optional): LSTM层数，默认为1。
            dropout (float, optional): dropout概率，默认为0.1。
        输入输出shape示例：
            输入: features.shape = [batch, seq_len, input_size]
            输出: logits.shape = [batch, num_labels]
        """
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
    def forward(self, features):
        """
        前向传播。
        参数：
            features (Tensor): 输入特征，形状为[batch, seq_len, input_size]
        返回：
            logits (Tensor): 分类logits，形状为[batch, num_labels]
        """
        output, _ = self.lstm(features)
        last_hidden = output[:, -1, :]
        return self.classifier(last_hidden)

# BiLSTM 分类头
class BiLSTMClassificationHead(nn.Module):
    def __init__(self, input_size, hidden_size, num_labels, num_layers=1, dropout=0.1):
        """
        双向LSTM分类头。
        适用于输入为序列特征的分类任务，能同时捕捉前向和后向信息。
        
        参数：
            input_size (int): 输入特征维度。
            hidden_size (int): LSTM隐藏层维度。
            num_labels (int): 分类类别数。
            num_layers (int, optional): LSTM层数，默认为1。
            dropout (float, optional): dropout概率，默认为0.1。
        输入输出shape示例：
            输入: features.shape = [batch, seq_len, input_size]
            输出: logits.shape = [batch, num_labels]
        """
        super().__init__()
        self.bilstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.classifier = nn.Linear(hidden_size * 2, num_labels)
    def forward(self, features):
        """
        前向传播。
        参数：
            features (Tensor): 输入特征，形状为[batch, seq_len, input_size]
        返回：
            logits (Tensor): 分类logits，形状为[batch, num_labels]
        """
        output, _ = self.bilstm(features)
        last_hidden = output[:, -1, :]  # [batch, hidden_size*2]
        return self.classifier(last_hidden)

classifier_map = {
    "mlp": MLPClassificationHead,
    "rnn": RNNClassificationHead,
    "lstm": LSTMClassificationHead,
    "bilstm": BiLSTMClassificationHead,
}