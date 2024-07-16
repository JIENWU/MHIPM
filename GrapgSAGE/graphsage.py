import pandas as pd
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import SAGEConv

# 定义GraphSAGE模型
class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats, out_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, 'mean')
        self.conv2 = SAGEConv(h_feats, out_feats, 'mean')

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h

# 从CSV文件中读取蛋白质对
def load_graph(csv_file):
    df = pd.read_csv(csv_file)
    src = df['protein1'].values - 1  # 调整索引为零基
    dst = df['protein2'].values - 1  # 调整索引为零基
    return dgl.graph((src, dst)), df

# 加载图和数据
csv_file = 'HBI pairs.csv'  # 替换为你的CSV文件路径
g, df = load_graph(csv_file)
g = dgl.to_bidirected(g)

# 初始化节点特征（随机特征）
num_nodes = g.num_nodes()
in_feats = 2  # 输入特征维度
h_feats = 4  # 隐层特征维度
out_feats = 512  # 输出特征维度
feat = torch.randn(num_nodes, in_feats)  # 使用随机特征初始化

# 实例化GraphSAGE模型
model = GraphSAGE(in_feats, h_feats, out_feats)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
num_epochs = 50

# 模型训练
for epoch in range(num_epochs):
    model.train()
    logits = model(g, feat)
    loss = F.mse_loss(logits, torch.ones(num_nodes, out_feats))  # 示例损失函数
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

# 提取嵌入向量
model.eval()
with torch.no_grad():
    embeddings = model(g, feat)

# 保存嵌入向量到CSV文件
protein_ids = list(range(1, num_nodes + 1))  # 生成蛋白质ID
embedding_df = pd.DataFrame(embeddings.numpy())
embedding_df = embedding_df.round(6)
embedding_df.insert(0, 'protein_id', protein_ids)
embedding_df.to_csv("HBI_graphsage_embeddings.csv", index=False)

print('Embeddings saved to HBI 图嵌入对比 graphsage embeddings.csv')
