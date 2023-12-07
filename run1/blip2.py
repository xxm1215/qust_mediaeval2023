import json
import os
import pickle

import torch
import pandas as pd
from PIL import Image
from lavis.models import load_model_and_preprocess
import clip
import tqdm

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# """"------------------文本特征提取----------------"""
# # rt_text_dataset = pd.read_table("./data/RT-Dataset/rt-test-text.txt")
# # rt_text_dataset.columns = rt_text_dataset.columns.str.strip()
# # text = rt_text_dataset[['url', "titleEN"]]
#
# P1_text_dataset = pd.read_table("./data/GDELT-Dataset/GDELT-P1-Test-Text.txt")
# P1_text_dataset.columns = P1_text_dataset.columns.str.strip()
# text = P1_text_dataset[['url', "text"]]
#
# # P2_text_dataset = pd.read_table("./data/GDELT-Dataset-2023-Part2-final/GDELT-P2-Test-Text.txt")
# # P2_text_dataset.columns = P2_text_dataset.columns.str.strip()
# # text = P2_text_dataset[['url', "title"]]
# # print(len(text['# article']))
#
# """-------------------------------------------------------"""
#
# """"--------------------------图片特征提取------------------------"""
# # rt_image_dataset = pd.read_table("./data/RT-Dataset/rt-test-img.txt")
# # rt_image_dataset.columns = rt_image_dataset.columns.str.strip()
# # image = rt_image_dataset["hashvalue"]
#
# P1_image_dataset = pd.read_table("./data/GDELT-Dataset/GDELT-P1-Test-Img.txt")
# P1_image_dataset.columns = P1_image_dataset.columns.str.strip()
# image = P1_image_dataset["imgFile"]
#
# # P2_image_dataset = pd.read_table("./data/GDELT-Dataset-2023-Part2-final/GDELT-P2-Test-Img.txt")
# # P2_image_dataset.columns = P2_image_dataset.columns.str.strip()
# # image = P2_image_dataset["imgFile"]
#
#
# # image = "./data/RT-Dataset/images-Test/" + image.str.strip()
# image = "./data/GDELT-Dataset/GDELT-P1-Test/" + image.str.strip()
# # image = "./data/GDELT-Dataset-2023-Part2-final/GDELT-P2-Test/" + image.str.strip()
#
# """-----------------------------------------------------------"""
# model, vis_processors, txt_processors = load_model_and_preprocess(
#     name="blip2_feature_extractor", model_type="pretrain", is_eval=True, device=device
# )
#
# hashvalue = {}
# texts = {}
#
# for i in image:
#     image = Image.open(i).convert('RGB')
#     image = vis_processors["eval"](image).unsqueeze(0).to(device)
#     image = {'image': image}
#     features_image = model.extract_features(image, mode="image")
#     # hashvalue[i.replace("./data/RT-Dataset/images-Test/", '')] = features_image.image_embeds_proj[:, 0, :]
#     hashvalue[i.replace("./data/GDELT-Dataset/GDELT-P1-Test/", '')] = features_image.image_embeds_proj[:, 0, :]
#     # hashvalue[i.replace("./data/GDELT-Dataset-2023-Part2-final/GDELT-P2-Test/", '')] = features_image.image_embeds_proj[:, 0, :]
#
# # with open('./data_pickle/RT-test-image-feature.pickle', 'wb') as pkl:
# # with open('./P1-test-image-feature.pickle', 'wb') as pkl:
# with open('./data_pickle/P1-test-image-feature.pickle', 'wb') as pkl:
#     pickle.dump(hashvalue, pkl)
#
#
# for index, line in text.iterrows():
#     # count = line['count']
#     count = line['url']
#
#     # caption = txt_processors['eval'](line['titleEN'])
#     caption = txt_processors['eval'](line['text'])
#     # caption = txt_processors['eval'](line['title'])
#
#     caption = {'text_input': [caption]}
#     features_text = model.extract_features(caption, mode="text").text_embeds_proj[:, 0, :]
#     texts[count] = features_text
# # with open('./RT-test-text-feature.pickle', 'wb') as pkl:
# with open('./data_pickle/P1-test-text-feature.pickle', 'wb') as pkl:
#     pickle.dump(texts, pkl)


import pandas as pd

import clip
import torch
import json
import os
import pickle
import tqdm


def process_text(model_name):
    BATCH_SIZE = 4

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载CLIP模型以及相应的预处理函数
    model, preprocess = clip.load(model_name, device=device)
    indexes = {}

    data = pd.read_table('./data/RT-Dataset/rt-test-text.txt')

    for index, item in data.iterrows():
        count = item["url"]  # 获取JSON文件中的"count"值
        text = item["text"]  # 获取JSON文件中的"text"值

    for i in tqdm.tqdm(range(0, len(data), BATCH_SIZE)):
        batch_data = data.iloc[i:i + BATCH_SIZE, :]
        batch_text = [item["text"] for _, item in batch_data.iterrows()]

        # 将文本数据截断为适合CLIP模型的长度
        context_length = 77  # 根据模型上下文长度进行调整
        batch_text = [text[:context_length] for text in batch_text]

        # 将文本数据转换为Tensor
        text_inputs = clip.tokenize(batch_text).to(device)

        with torch.no_grad():
            text_features = model.encode_text(text_inputs).cpu().numpy()
            # print(text_features)

        for index, item in batch_data.iterrows():
            count = item["url"]
            indexes[count] = text_features.tolist()

    # 保存到pickle文件中
    with open('./clip/clip_text_features.pickle', 'wb') as f:
        pickle.dump(indexes, f)


if __name__ == '__main__':
    model_name = 'ViT-L/14'  # 指定要使用的CLIP模型
    process_text(model_name)
