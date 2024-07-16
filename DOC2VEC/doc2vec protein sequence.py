import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# 从CSV文件中读取蛋白质名字和序列
df = pd.read_csv("PBI sequence.csv")

# 获取蛋白质名字和序列
protein_names = df['Protein Name'].tolist()
protein_sequences = df['Protein Sequence'].tolist()

# 将每个蛋白质序列视为一个文档，每个氨基酸视为一个词
documents = [TaggedDocument(words=list(seq), tags=[name]) for name, seq in zip(protein_names, protein_sequences)]

# 定义Doc2Vec模型
model = Doc2Vec(size=64, min_count=5, window=5, workers=4, dm=1)

# 构建词汇表
model.build_vocab(documents)

# 训练Doc2Vec模型
model.train(documents, total_examples=model.corpus_count, epochs=100)


# 获取每个蛋白质序列的向量表示
protein_vectors = [model.docvecs[name] for name in protein_names]


# 将向量保存为CSV文件，并控制小数点后显示四位
output_df = pd.DataFrame(protein_vectors, columns=[f"feature_{i}" for i in range(64)])
output_df = output_df.round(6)
output_df.insert(0, 'Protein Name', protein_names)
output_df.to_csv("PBI_doc2vec_protein_vectors.csv", index=False)

# 打印示例向量
print(output_df.head())
