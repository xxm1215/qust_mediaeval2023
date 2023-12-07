import torch
import clip
from PIL import Image
import os
import numpy as np
import pickle
import argparse
import tqdm



def process(model, model_name):
    BATCH_SIZE = 4

    image_extensions = ['.jpg']

    path = '/home/qd/PycharmProjects/MediaEval2022/rt1000_images'

    image_paths = []

    # 遍历指定目录（path）下的所有文件和子目录
    for file in os.listdir(path):
        filepath = os.path.join(path, file)
        if not os.path.isfile(filepath):
            continue
        # 遍历定义在image_extensions列表中的图像文件扩展名，例如'.jpg'
        for suffix in image_extensions:
            if file.endswith(suffix):
                image_paths.append(filepath)
                break

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载CLIP模型以及相应的预处理函数
    model, preprocess = clip.load(model, device=device)
    indexes = {}

    if os.path.exists('image_features336.pickle'):
        with open('image_features336.pickle', 'rb') as f:
            indexes = pickle.load(f)

    for i in tqdm.tqdm(range(0, len(image_paths), BATCH_SIZE)):
        mb_paths = image_paths[i:i + BATCH_SIZE]
        images = []
        corrupted_images = []
        for path in mb_paths:
            try:
                image = preprocess(Image.open(path))
                images.append(image)
            except Exception as e:
                corrupted_images.append(path)

        for corrupted_img in corrupted_images:
            mb_paths.remove(corrupted_img)

        images = torch.stack(images).to(device)

        with torch.no_grad():
            image_features = model.encode_image(images).cpu().numpy()

        for j, path in enumerate(mb_paths):
            indexes[path] = image_features[j].tolist()

    with open('image_features336.pickle', 'wb') as f:   # 使用
        pickle.dump(indexes, f)


if __name__ == '__main__':
    models = [
        # 'RN50',
        # 'RN101',
        # 'RN50x4',
        # 'RN50x16',
        # 'RN50x64',
        # 'ViT-B/32',
        # 'ViT-B/16',
        # 'ViT-L/14',
        'ViT-L/14@336px'
    ]
    model_names = [
        # 'RN50',
        # 'RN101',
        # 'RN50x4',
        # 'RN50x16',
        # 'RN50x64',
        # 'ViT-B-32',
        # 'ViT-B-16',
        # 'ViT-L-14',
        'ViT-L-14@336px'
    ]
    for i, model in enumerate(models):
        process(model, model_names[i])


