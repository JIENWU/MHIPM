import numpy as np
import pandas as pd

# 读取两个CSV文件，假设没有列名
file1 = pd.read_csv('PBI_ESM_feature.csv', header=None)
file2 = pd.read_csv('PBI_doc2vec_protein_vectors.csv', header=None)

# 提取标签和特征向量
labels = file1.iloc[:, 0].values
features1 = file1.iloc[:, 1:].values
features2 = file2.iloc[:, 1:].values

# 计算特征向量之间的相似度
def calculate_similarity(feature1, feature2):
    return np.dot(feature1, feature2) / (np.linalg.norm(feature1) * np.linalg.norm(feature2))

# 计算注意力权重
def calculate_attention_weights(features1, features2):
    weights = np.zeros((len(features1), len(features2)))
    for i, f1 in enumerate(features1):
        for j, f2 in enumerate(features2):
            weights[i][j] = calculate_similarity(f1, f2)
    return weights

# 加权求和
def weighted_sum(features, weights):
    return np.dot(weights, features)

# 计算注意力权重
attention_weights = calculate_attention_weights(features1, features2)

# 融合特征向量
merged_features = weighted_sum(features2, attention_weights)

# 合并标签和融合后的特征向量
merged_data = np.column_stack((labels, merged_features))

# 将融合后的特征向量和标签保存为CSV文件
merged_df = pd.DataFrame(merged_data)
merged_df = merged_df.round(6)
merged_df.to_csv('PBI_融合特征_attention.csv', index=False, header=None)
