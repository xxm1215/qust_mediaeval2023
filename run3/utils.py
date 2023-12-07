import torch
import pandas as pd
import json
import random
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, Subset
import clip

from PIL import Image

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


class MediaEval24_Dataset(Dataset):
    def __init__(self, path):
        with open(path,'r') as inf:
            self.data = json.load(inf)

        self.labels = [l['label'] for l in self.data]
        # ViT-L/14@336px， ViT-B/32
        self.clip_model, self.preprocess = clip.load("ViT-L/14@336px", device=device, jit=False)  # Best model use ViT-B/32

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        txt = self.data[idx]['text']
        img = self.data[idx]['image']

        img = self.preprocess(Image.open("/mnt/qust_521_big_2/public/MediaEval2023/GDELT-Dataset-2023-Part2/GDELT-Dataset-2023-Part2-final/GDELT-P2-Training/" + img)).unsqueeze(0).to(device)
        txt = clip.tokenize(txt, truncate=True).to(device)
        image_embedding = self.clip_model.encode_image(img)
        text_embedding = self.clip_model.encode_text(txt)

        label = self.data[idx]['label']
        # print(image_embedding.size()) #[1, 512]
        # print(text_embedding.size()) #[1 512]
        return text_embedding.squeeze(), image_embedding.squeeze(), torch.tensor(label)


class MediaEval24_Test_Dataset(Dataset):
    def __init__(self, path):
        with open(path,'r') as inf:
            self.data = json.load(inf)

        # ViT-L/14@336px， ViT-B/32
        self.clip_model, self.preprocess = clip.load("ViT-L/14@336px", device=device, jit=False)  # Best model use ViT-B/32

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        txt = self.data[idx]['text']
        img1 = self.data[idx]['image']

        img = self.preprocess(Image.open("/mnt/qust_521_big_2/public/MediaEval2023/GDELT-Dataset-2023-Part2/GDELT-Dataset-2023-Part2-final/GDELT-P2-Test/" + img1)).unsqueeze(0).to(device)

        txt = clip.tokenize(txt, truncate=True).to(device)
        image_embedding = self.clip_model.encode_image(img)
        text_embedding = self.clip_model.encode_text(txt)

        url = self.data[idx]['text_url']
        # print(image_embedding.size()) #[1, 512]
        # print(text_embedding.size()) #[1 512]
        return text_embedding.squeeze(), image_embedding.squeeze(), url, img1

# train_dataset = MediaEval24_Dataset("./rt-train-textimg.json")
#
# next(iter(train_dataset))

def train_json_builder(txt_file, m):
    with open(txt_file, 'r') as f:
        lines = f.readlines()
    
    print(len(lines))
    
    images = []
    title_en = []
    texts_en = []

    header = lines[0].strip().split()
    img_index = header.index("imgFile") # imgFile
    texts_en_index = header.index("text")
    title_en_index = header.index("title")
    # print(header)
    # print(texts_en_index)
    # print(title_en_index)
    # print(img_index)
    for line in lines[1:]:
        data = line.strip().split('\t')
        if len(data) >= max(img_index, texts_en_index, title_en_index) + 1:
            img = data[8]
            title = data[6]
            # print(title)
            text = data[7]
            # print(text)
            images.append(img)
            title_en.append(title)
            texts_en.append(text)


    dataset = [{"image": img, "text": title+text, "label": 1} for img, title, text in zip(images, title_en, texts_en)]

    total_size = len(images)
    for i, img in tqdm(enumerate(images), total=total_size):
        non_matching_titles = title_en[:i] + title_en[i + 1:]
        non_matching_texts = texts_en[:i] + texts_en[i + 1:]
        num_to_sample = int(m * (total_size-1))
        combined_lists = list(zip(non_matching_titles, non_matching_texts))
        sampled_elements = random.sample(combined_lists, num_to_sample)
        sampled_titles, sampled_texts = zip(*sampled_elements)

        for i in range(len(sampled_texts)):
            dataset.append({"image": img, "text": sampled_titles[i]+sampled_texts[i], "label": 0})
            #print({"image": img, "text": sampled_titles[i]+sampled_texts[i], "label": 0})

    with open('./GDELT_dataset_p2_0.0004.json', 'w') as f:
        json.dump(dataset, f, indent=4)

    print("Done!")


def test_json_builder(img_info_file, txt_info_file, m):
    
    with open(txt_info_file, 'r') as f1:
        text_lines = f1.readlines()
        
    with open(img_info_file, 'r') as f2:
        img_lines = f2.readlines()
    
    print(len(text_lines))
    print(len(img_lines))
    images = []
    title_en = []
    texts_en = []
    urls = []

    img_header = img_lines[0].strip().split()
    print(img_header)
    img_index = img_header.index("imgFile")


    text_header = text_lines[0].strip().split()
    print(text_header)
    texts_en_index = text_header.index("text")
    title_en_index = text_header.index("title")
    url_index = text_header.index("url")

    print(img_index) # 12
    print(texts_en_index) # 14
    print(title_en_index) # 13
    print(url_index) # 2
    #for img_line in img_lines[1:]:
    #    img_data = img_line.strip().split('\t')
        #print(img_data[img_index])
        #print(len(img_data))
        #if len(img_data) >= max(img_index, texts_en_index, title_en_index) + 1:
    #    img = img_data[img_index]
            #print(img)
    #    images.append(img)
            
    #for text_line in text_lines[1:]:
    #    text_data = text_line.strip().split('\t')
        # print(data)
    #    if len(text_data) >= max(img_index, texts_en_index, title_en_index) + 1:
    #         title = text_data[title_en_index]
            #print(title)
    #         text = text_data[texts_en_index]
            #print(text)
    #         title_en.append(title)
    #         texts_en.append(text)
    
    for idx in range(len(img_lines)):
        img_data = img_lines[idx].strip().split('\t')
        text_data = text_lines[idx].strip().split('\t')
        if len(text_data) >= max(img_index, texts_en_index, title_en_index) + 1:
             title = text_data[7]
             text = text_data[6]
             img = img_data[4]
             # print(img)
             url = text_data[2]
             title_en.append(title)
             texts_en.append(text)
             images.append(img)
             urls.append(url)
             



    print(len(images))
    print(len(texts_en))
    print(len(title_en))

    title_en.pop(0)
    texts_en.pop(0)
    images.pop(0)
    urls.pop(0)

    dataset1 = []
    total_size = len(images)
    print(total_size)
    for i, img1 in tqdm(enumerate(images), total=total_size):
        non_matching_titles = title_en[:i] + title_en[i + 1:]
        non_matching_texts = texts_en[:i] + texts_en[i + 1:]
        non_matching_urls = urls[:i] + urls[i + 1:]
        num_to_sample = int(m * (total_size-1))
        combined_lists = list(zip(non_matching_titles, non_matching_texts, non_matching_urls))
        sampled_elements = random.sample(combined_lists, num_to_sample)
        sampled_titles, sampled_texts, sampled_urls = zip(*sampled_elements)


        for i in range(len(sampled_texts)):
            # print({"image": img1, "text": sampled_titles[i]+sampled_texts[i], "text_url": sampled_urls[i]})
            new_dict = {}

            new_dict["image"] = img1
            new_dict["text"] = sampled_titles[i] + " " + sampled_texts[i]
            new_dict["text_url"] = sampled_urls[i]
            dataset1.append(new_dict)
            # dataset1.append({"image": img, "text": sampled_titles[i]+sampled_texts[i], "text_url": sampled_urls[i]})
            # print({"image": img1, "text": sampled_titles[i]+sampled_texts[i], "text_url": sampled_urls[i]})

    with open('./GDELT_test_p2_dataset.json', 'w') as outf:
        print('strat to dump!')
        json.dump(obj=dataset1, fp=outf, indent=4)

    print("Done!")

# train_json_builder("/mnt/qust_521_big_2/public/MediaEval2023/GDELT-Dataset-2023-Part2/GDELT-Dataset-2023-Part2-final/GDELT-P2-Training.txt", 0.0004)
# test_json_builder('/mnt/qust_521_big_2/public/MediaEval2023/GDELT-Dataset-2023-Part2/GDELT-Dataset-2023-Part2-final/GDELT-P2-Test-Img.txt',
#                  '/mnt/qust_521_big_2/public/MediaEval2023/GDELT-Dataset-2023-Part2/GDELT-Dataset-2023-Part2-final/GDELT-P2-Test-Text.txt',
#                  1.0)

# with open('./rt_test_dataset.json', 'r') as inf:
#     data = json.load(inf)
#     for idx, line in enumerate(data):
#         print(line)
#         if idx % 2990 == 0 :
#             print("="*100)
