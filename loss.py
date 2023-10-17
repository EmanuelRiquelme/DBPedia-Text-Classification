import torch
import torch.nn as nn

class Loss(nn.Module):
  def __init__(self):
        super(Loss, self).__init__()
        self.loss = nn.CrossEntropyLoss()

  def forward(self,pred,labels):
    """
    l1,l2,l3 = 9,70,219
    """
    loss = self.loss(pred[...,:9],labels[...,0])/9
    loss += self.loss(pred[...,9:79],labels[...,1])/70
    loss = self.loss(pred[...,-219:],labels[...,2])/219
    return loss
