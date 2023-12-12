# coding=utf-8
import os

import numpy as np
import torch
from sklearn.manifold import TSNE
from transformers import BertTokenizer, AutoConfig, BertForMaskedLM, LineByLineTextDataset, \
    DataCollatorForLanguageModeling, \
    Trainer, TrainingArguments, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-ancient-chinese')

# test = tokenizer.encode('善', return_tensors='pt', max_length=512, padding='max_length', truncation=True)
# print(test)
# id = test.cpu().detach().numpy()[0][1]
# print(id)
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')


def reduce_dimensions(vectors, labels, num_dimensions=2):
    # extract the words & their vectors, as numpy arrays
    vectors = np.asarray(vectors)
    labels = np.asarray(labels)  # fixed-width numpy strings

    # reduce using t-SNE，一定要指定随机种子，这样每次降维后的结果才能一样
    tsne = TSNE(n_components=num_dimensions, random_state=42, learning_rate='auto', n_iter=1000)
    vectors = tsne.fit_transform(vectors)

    return vectors


def plot_word2vec(vectors_tsne, labels, file_name, is_show=False, is_save="True"):
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
    plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题

    plt.figure(figsize=(12, 12), dpi=300)
    plt.scatter(vectors_tsne[:, 0], vectors_tsne[:, 1])
    for i, word in enumerate(labels):
        x, y = vectors_tsne[i]
        plt.annotate(word, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')

    # 标注文本
    # for i in indices:
    #     plt.scatter(x_vals[i], y_vals[i],words[i],
    #              fontdict={
    #                         # 'family': '宋体', # 标注文本字体
    #                         'fontsize': 10, # 文本大小
    #                         # 'fontweight': 'bold', # 字体粗细
    #                         # 'fontstyle': 'italic', # 字体风格
    #                         'color': "black",  # 文本颜色
    #                  },
    #
    #              )
    # Label randomly subsampled 25 data points
    # selected_indices = random.sample(indices, 25)
    # for i in selected_indices:
    #     plt.annotate(words[i], (x_vals[i], y_vals[i]))
    if is_show:
        plt.show()
    if is_save:
        plt.savefig(f"./{file_name}.png")


from scipy.linalg import orthogonal_procrustes


def align_word_vectors(base_matrix, other_matrices):
    """
    将其他时间段的词向量矩阵对齐到基准时间段。
    参数：
    - base_matrix: numpy数组，基准时间段的词向量矩阵，形状为(num_words, num_dims)。
    - other_matrices: numpy数组列表，其他时间段的词向量矩阵列表，形状均为(num_words, num_dims)。
    返回：
    - numpy数组列表，对齐后的词向量矩阵列表，形状均为(num_words, num_dims)。
    """

    # 对每个词向量矩阵进行标准化（去中心化和缩放）
    def normalize(matrix):
        return (matrix - matrix.mean(axis=0)) / matrix.std(axis=0)

    base_norm = normalize(base_matrix)
    others_norm = [normalize(matrix) for matrix in other_matrices]

    # 将其他时间段的词向量矩阵对齐到基准时间段
    aligned_matrices = []
    for i in range(len(others_norm)):
        M, _ = orthogonal_procrustes(base_norm, others_norm[i])
        aligned_matrix = other_matrices[i].dot(M)
        aligned_matrices.append(aligned_matrix)

    return aligned_matrices


def test_emebdding():
    from sklearn.metrics.pairwise import cosine_similarity

    file_list = [item.split('.')[0] for item in os.listdir('./single_train_weight/')]


    embedding_dict = {}
    mean_embedding = []
    for i in file_list:
        name = i.split('.')[0]
        embedding_weight = torch.load(f'./single_train_weight/{name}.pt', map_location='cpu')

        embedding_weight.requires_grad = False

        embedding_dict[name] = embedding_weight
        mean_embedding.append(embedding_weight)

    mean_embedding = torch.stack(mean_embedding,dim=0).mean(dim=0)

    base_matric = np.array(mean_embedding.detach())
    other_matrices = []
    # embedding_weight = torch.load(f'./embedding_weight/东汉.pt', map_location='cpu')
    #
    # embedding_weight.requires_grad = False
    # embedding_dict['东汉'] = embedding_weight
    for i in embedding_dict:
        other_matrices.append(embedding_dict[i].detach().numpy())
    other_matrices = np.array(other_matrices)
    aligned_embeddings = align_word_vectors(base_matric, other_matrices)
    new_embedding_dict = {}
    for i, j in zip(file_list, aligned_embeddings):
        new_embedding_dict[i] = j
        tmp = torch.from_numpy(j)
        torch.save(tmp, f'./single_train_weight/mean/{i}.pth')
    print('保存成功')
    # compass_weight = mean_embedding[id].tolist()
    # for i in file_list:
    #     print(i)
    #     s = cosine_similarity([new_embedding_dict[i][id].tolist()], [compass_weight])
    #     print(s)


test_emebdding()

# 东汉
# [[0.05779161]]
# 南朝宋
# [[0.59192261]]
# 西晋
# [[0.67087923]]
# 唐
# [[0.72758629]]
# 南朝梁
# [[0.69681881]]
# 北朝齐
# [[0.36065565]]
# 后晋
# [[0.67591968]]
# 宋
# [[0.76312208]]
# 元
# [[0.20362289]]
# 明
# [[0.7755116]]
# 清
# [[0.73301991]]
# 西汉
# [[0.61632725]]

