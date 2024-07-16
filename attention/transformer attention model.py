import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 读取两个CSV文件，假设没有列名
file1 = pd.read_csv('HBI ESM vector.csv', header=None)
file2 = pd.read_csv('HBI_doc2vec_protein_vectors.csv', header=None)

# 提取标签和特征向量
labels = file1.iloc[:, 0].values
features1 = file1.iloc[:, 1:].values
features2 = file2.iloc[:, 1:].values

# 转换为PyTorch张量
features1_tensor = torch.tensor(features1, dtype=torch.float32)
features2_tensor = torch.tensor(features2, dtype=torch.float32)

# 定义自注意力机制
class SelfAttention(nn.Module):
    def __init__(self, feature_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.scale = feature_dim ** 0.5

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attention_weights = torch.softmax(torch.matmul(Q, K.transpose(-2, -1)) / self.scale, dim=-1)
        out = torch.matmul(attention_weights, V)
        return out

# 定义融合特征的模型
class FeatureFusion(nn.Module):
    def __init__(self, feature_dim):
        super(FeatureFusion, self).__init__()
        self.attention = SelfAttention(feature_dim)

    def forward(self, features1, features2):
        concatenated_features = torch.cat((features1, features2), dim=1)
        fused_features = self.attention(concatenated_features)
        return fused_features

# 创建模型实例
feature_dim = features1.shape[1] + features2.shape[1]
model = FeatureFusion(feature_dim)

# 准备数据
dataset = TensorDataset(features1_tensor, features2_tensor)
dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

# 融合特征
fused_features_list = []
model.eval()
with torch.no_grad():
    for batch_features1, batch_features2 in dataloader:
        fused_features = model(batch_features1, batch_features2)
        fused_features_list.append(fused_features)

# 合并所有融合后的特征
fused_features = torch.cat(fused_features_list, dim=0).numpy()

# 合并标签和融合后的特征向量
merged_data = np.column_stack((labels, fused_features))

# 将融合后的特征向量和标签保存为CSV文件
merged_df = pd.DataFrame(merged_data)
merged_df = merged_df.round(6)
merged_df.to_csv('HBI_transofrmer_融合特征_attention_256.csv', index=False, header=None)
