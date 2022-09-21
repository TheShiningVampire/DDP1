import torch
import torch.nn as nn
import torch.nn.functional as F


# Define the Contrastive Loss Function
class ContrastiveLoss(torch.nn.Module):
    def __init__(self,
                margin: float = 2.0,
                ):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
      # Calculate the euclidian distance and calculate the contrastive loss
      # euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)

      # Find cosine distance between the two vectors
      cosine_similarity = F.cosine_similarity(output1, output2, dim=1, eps=1e-6)
      cosine_distance = 1 - cosine_similarity

      loss_contrastive = torch.mean((1-label) * torch.pow(cosine_distance, 2) +
                                    (label) * torch.pow(torch.clamp(self.margin - cosine_distance, min=0.0), 2))


      return loss_contrastive
