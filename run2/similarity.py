# import pickle
# import numpy as np
# import json
# from sklearn.metrics.pairwise import cosine_similarity

# def load_features(pickle_path):
#     with open(pickle_path, 'rb') as f:
#         return pickle.load(f)

# def get_feature_matrix(feature_dict):
#     return np.array(list(feature_dict.values()))

# def calculate_cosine_similarity(text_features, image_features):
#     similarity_matrix = cosine_similarity(text_features, image_features)
#     return similarity_matrix

# def rank_similarities(similarity_matrix, text_keys, image_keys, top_n=100):
#     rankings = {}
#     for i, similarities in enumerate(similarity_matrix):
#         ranked_indices = np.argsort(-similarities)[:top_n]  # 仅获取前100个
#         rankings[text_keys[i]] = [image_keys[idx] for idx in ranked_indices]
#     return rankings

# # 加载特征
# text_features = load_features('text_features.pickle')
# image_features = load_features('image_features.pickle')

# # 获取特征矩阵
# text_feature_matrix = get_feature_matrix(text_features)
# image_feature_matrix = get_feature_matrix(image_features)

# # 计算余弦相似度
# similarity_matrix = calculate_cosine_similarity(text_feature_matrix, image_feature_matrix)

# # 对相似度进行排名，仅保留前100个结果
# rankings = rank_similarities(similarity_matrix, list(text_features.keys()), list(image_features.keys()))

# # 将结果保存到JSON文件
# with open('top_100_text_image_similarity_rankings.json', 'w') as json_file:
#     json.dump(rankings, json_file)

# print("Top 100 rankings saved to top_100_text_image_similarity_rankings.json")



import pickle
import numpy as np
import json
import torch
from sklearn.metrics.pairwise import cosine_similarity

def load_features(pickle_path):
    with open(pickle_path, 'rb') as f:
        return pickle.load(f)

def get_feature_matrix(feature_dict):
    return np.array(list(feature_dict.values()))

def calculate_cosine_similarity(text_features, image_features):
    similarity_matrix = cosine_similarity(text_features, image_features)
    return similarity_matrix

# Define the dual softmax function
def dual_softmax_pt(similarity_matrix):
    ss_row = torch.nn.Softmax(dim=0)
    m = ss_row(torch.tensor(similarity_matrix))
    ss_col = torch.nn.Softmax(dim=1)
    n = ss_col(torch.tensor(similarity_matrix))
    c = m * n
    return c.data.numpy()

def rank_similarities(similarity_matrix, text_keys, image_keys, top_n=100):
    rankings = {}
    for i, similarities in enumerate(similarity_matrix):
        ranked_indices = np.argsort(-similarities)[:top_n]  # 仅获取前100个
        ranked_paths = [image_keys[idx] for idx in ranked_indices]
        
        # 去掉路径中的前缀
        ranked_paths = [path.replace('/home/qd/PycharmProjects/MediaEval2022/rt1000_images/', '') for path in ranked_paths]
        
        rankings[text_keys[i]] = ranked_paths
    return rankings

# 加载特征
text_features = load_features('text_features336.pickle')
image_features = load_features('image_features336.pickle')

# 获取特征矩阵
text_feature_matrix = get_feature_matrix(text_features)
image_feature_matrix = get_feature_matrix(image_features)

# 计算余弦相似度
similarity_matrix = calculate_cosine_similarity(text_feature_matrix, image_feature_matrix)

# Apply dual softmax to similarity matrix
dual_softmax_result = dual_softmax_pt(similarity_matrix)

print(dual_softmax_pt)
# 对相似度进行排名，仅保留前100个结果
rankings = rank_similarities(dual_softmax_result, list(text_features.keys()), list(image_features.keys()))
# print(rankings)
# 将结果保存到JSON文件
with open('rt1000_similarity_rankings336.json', 'w') as json_file:
    json.dump(rankings, json_file)

print("Top 100 rankings saved to top_100_text_image_similarity_rankings.json")

