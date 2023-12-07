import torch.nn.functional as F
from model import *
from utils import *
import clip
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import BinaryAccuracy
import torch
import gc
# from torchvision.ops import sigmoid_focal_loss 


writer = SummaryWriter()
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)
model = SiameseNetwork()
model.to(device)
dataset = MediaEval24_Dataset("./GDELT_dataset_p2_0.0004.json")
y = dataset.labels
print(len(y))


EPOCH = 3

best_val_acc = 0
iteration_number = 1
train_loader = DataLoader(dataset, batch_size=28, shuffle=True)


metric = BinaryAccuracy().to(device)
criterion1 = ContrastiveLoss()
criterion3 = torch.nn.BCEWithLogitsLoss()

optimizer = optim.AdamW(model.parameters(), lr=5e-6, betas=(0.9, 0.98), eps=1e-6,
                        weight_decay=0.05)
p = 0.8

for epoch in range(EPOCH):
    for txt, img, label in tqdm(train_loader):
        iteration_number += 1
        txt, img, label = txt.to(device, torch.float32), img.to(device, torch.float32), label.to(device, torch.float32)
        # Zero the gradients
        optimizer.zero_grad()

        # Pass in the two images into the network and obtain two outputs
        output1, output2, output3 = model(txt, img)

        #sim_score = cos(output1, output2)

        # Pass the outputs of the networks and label into the loss function
        loss_contrastive = criterion1(output1, output2, label)
        # loss_cossim = criterion2(output1, output2, label)

        loss_crossent = criterion3(output3, label)
        #loss_focal = sigmoid_focal_loss(output3, label, reduction='sum')

        loss = p * loss_contrastive + (1-p) * loss_crossent 

        # Calculate the backpropagation
        #loss_contrastive.backward()

        loss.backward()
        # Optimize
        optimizer.step()
        #scheduler.step(loss)

        writer.add_scalar("total_loss/train",  loss.detach().item(), iteration_number)
        writer.add_scalar("contrastive_loss/train",loss_contrastive.detach().item(), iteration_number)
        # writer.add_scalar("fold{}_cosin_loss/train".format(idx+1),loss_cossim.detach().item(), iteration_number)
        writer.add_scalar("bce_loss/train", loss_crossent.detach().item(), iteration_number)
        # writer.add_scalar("fold{}_focal_loss/train".format(idx+1), loss_focal.detach().item(), iteration_number)
        torch.save(model.state_dict(), './model_GDELT_p2_{}@{}.pt'.format(epoch+1, "rt"))

            #if iteration_number % 5000 == 0:
       # with torch.no_grad():
       #     correct = 0
       #     total = 0
       #     total_acc = 0
       #     model.eval()
       #     for eval_txt, eval_img, eval_label in tqdm(val_loader):
       #         eval_txt, eval_img, eval_label = eval_txt.to(device, torch.float32), eval_img.to(device, torch.float32), eval_label.to(device, torch.float32)
       #         eval_out1, eval_out2, eval_out3 = model(eval_txt, eval_img)
                #eval_sim_score = cos(eval_out1, eval_out2)
                #eval_sim_score -= eval_sim_score.min()
                #eval_sim_score /= eval_sim_score.max()
                #eval_sim_score = torch.where(eval_sim_score > 0.5, 1.0, 0.0)

       #         #euclidean_distance = F.pairwise_distance(eval_out1, eval_out2)
       #         eval_acc = metric(eval_out3, eval_label)
       #         total_acc += eval_acc
       #         total += 1
                #if total == 10:
                #    break
       #     print("Epoch {}  Eval_acc: {} ".format(epoch, total_acc/total))
       #     writer.add_scalar("fold{}_Acc/eval".format(idx+1), total_acc/total, iteration_number)
       #     if total_acc/total > best_val_acc:
       #         best_val_acc = total_acc/total
       #         torch.save(model.state_dict(), './model_vacc_{}@{}.pt'.format(round(float(best_val_acc), 5), idx+1))
       #     gc.collect()


