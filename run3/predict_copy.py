import json

from model import *
from utils import *
import clip
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import BinaryAccuracy
import torch

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)
model = SiameseNetwork()
model.load_state_dict(torch.load("model_GDELT_p2_3@rt.pt"))
model.to(device)
dataset = MediaEval24_Test_Dataset("./GDELT_test_p2_dataset.json")
test_loader = DataLoader(dataset, batch_size=16)
crit = nn.Sigmoid()
model.eval()

predictions = []
c = 0
with torch.no_grad():
    for txt, img, t_idx, i_idx in tqdm(test_loader):
        txt, img = txt.to(device, torch.float32), img.to(device, torch.float32)

        output1, output2, output3 = model(txt, img)

        cos = torch.nn.CosineSimilarity(dim=1)
        sim_score = cos(output1, output2)

        pred = crit(output3)

        t_idx = [t for t in t_idx]
        # print(t_idx)
        i_idx = [i for i in i_idx]
        # print(i_idx)
        sim_score = sim_score.cpu().tolist()
        pred = pred.cpu().tolist()

        for idx in range(len(t_idx)):
            new_dict = {}
            url = t_idx[idx]
            img_name = i_idx[idx]
            ss = sim_score[idx]
            if pred[idx] >= 0.5:
                cls = 1
            else:
                cls = 0
            # print(img_name)
            new_dict['url'] = url
            new_dict['img_file_name'] = img_name.split(".jpg")[0]
            new_dict['cos_sim'] = ss
            new_dict['binary_cls'] = cls
            # print(new_dict)
            c += 1
            with open("/mnt/qust_521_big_2/public/MediaEval2023/predictions/P2/{}_{}.json".format(img_name, c + 1),
                    'w') as out_f:
                json.dump(new_dict, out_f, indent=4)
