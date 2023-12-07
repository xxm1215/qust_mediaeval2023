import clip
import torch
import json
import os
import pickle
import tqdm


# def process_text(json_file_path, model_name):
#     BATCH_SIZE = 4

#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     # 加载CLIP模型以及相应的预处理函数
#     model, preprocess = clip.load(model_name, device=device)
#     indexes = {}

#     if os.path.exists('text_features.pickle'):
#         with open('text_features.pickle', 'rb') as f:
#             indexes = pickle.load(f)

#     with open(json_file_path, 'r', encoding='utf-8') as json_file:
#         data = json.load(json_file)

#     text_data = [item["text"] for item in data]  # 提取每个JSON对象中的"text"字段
#     # count = [item["count"] for item in data]

#     for i in tqdm.tqdm(range(0, len(text_data), BATCH_SIZE)):
#         batch_text = text_data[i:i + BATCH_SIZE]

#         # 将文本数据截断为适合CLIP模型的长度
#         context_length = 77  # 根据模型上下文长度进行调整
#         batch_text = [text[:context_length] for text in batch_text]

#         # 将文本数据转换为Tensor
#         text_inputs = clip.tokenize(batch_text).to(device)

#         with torch.no_grad():
#             text_features = model.encode_text(text_inputs).cpu().numpy()

#         for j, text in enumerate(batch_text):
#             indexes[text] = text_features[j].tolist()

#     with open('text_features.pickle', 'wb') as f:
#         pickle.dump(indexes, f)



def process_text(json_file_path, model_name):
    BATCH_SIZE = 4

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载CLIP模型以及相应的预处理函数
    model, preprocess = clip.load(model_name, device=device)
    indexes = {}

    if os.path.exists('text_features336.pickle'):
        with open('text_features336.pickle', 'rb') as f:
            indexes = pickle.load(f)

    with open(json_file_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)

    for item in data:
        count = item["count"]  # 获取JSON文件中的"count"值
        text = item["text"]  # 获取JSON文件中的"text"值

        # 将"count"值作为键，文本作为值，存储在indexes字典中
        indexes[count] = text

    for i in tqdm.tqdm(range(0, len(data), BATCH_SIZE)):
        batch_data = data[i:i + BATCH_SIZE]
        batch_text = [item["text"] for item in batch_data]

        # 将文本数据截断为适合CLIP模型的长度
        context_length = 77  # 根据模型上下文长度进行调整
        batch_text = [text[:context_length] for text in batch_text]

        # 将文本数据转换为Tensor
        text_inputs = clip.tokenize(batch_text).to(device)

        with torch.no_grad():
            text_features = model.encode_text(text_inputs).cpu().numpy()

        for j, item in enumerate(batch_data):
            count = item["count"]
            indexes[count] = text_features[j].tolist()

    # 保存到pickle文件中
    with open('text_features336.pickle', 'wb') as f:
        pickle.dump(indexes, f)


if __name__ == '__main__':
    model_name = 'ViT-L/14@336px'  # 指定要使用的CLIP模型
    # json_file_path = 'test_new.json'  # 替换为您的JSON文件路径
    json_file_path = 'test_new.json'  # 替换为您的JSON文件路径
    
    process_text(json_file_path, model_name)
