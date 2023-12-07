import torch
import torch.nn as nn
import torch.nn.functional as F



# Define the Contrastive Loss Function
class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
      # Calculate the euclidean distance and calculate the contrastive loss
      euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)

      loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                    (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

      return loss_contrastive


class SiameseNetwork(nn.Module):

    def __init__(self):
        super(SiameseNetwork, self).__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=8, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

        self.self_att = nn.MultiheadAttention(embed_dim=768, num_heads=8, dropout=.5, batch_first=True)

        # Setting up the Fully Connected Layers
        self.fc1 = nn.Sequential(
            nn.Linear(768, 1536),
            nn.ELU(inplace=True),

            nn.Linear(1536, 768),
            nn.ELU(inplace=True),

            nn.Linear(768, 512)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(1536, 768),
            nn.ELU(inplace=True),

            nn.Linear(768, 32),
            nn.ELU(inplace=True),

            nn.Linear(32, 1)
        )


    def forward(self, txt, img):
        # Its output is used to determine the similiarity
        #t_output = self.transformer_encoder(txt)
        #i_output = self.transformer_encoder(img)
        t_proj = self.fc1(txt)
        i_proj = self.fc1(img)

        t2i, _ = self.self_att(img, img, img)
        i2t, _ = self.self_att(txt, txt, txt)

        all_out = torch.stack((t2i, i2t), dim=1)
        all_out_std, all_out_mean = torch.std_mean(all_out, dim=1)

        all_out = torch.cat((all_out_std, all_out_mean), dim=1) # 1536 
        cls_out = self.fc2(all_out)

        return t_proj, i_proj, cls_out.squeeze()


