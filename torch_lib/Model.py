import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import repvgg_pytorch as repvgg

def OrientationLoss(orient_batch, orientGT_batch, confGT_batch):

    batch_size = orient_batch.size()[0]
    indexes = torch.max(confGT_batch, dim=1)[1]

    # extract just the important bin
    orientGT_batch = orientGT_batch[torch.arange(batch_size), indexes]
    orient_batch = orient_batch[torch.arange(batch_size), indexes]

    theta_diff = torch.atan2(orientGT_batch[:,1], orientGT_batch[:,0])
    estimated_theta_diff = torch.atan2(orient_batch[:,1], orient_batch[:,0])

    return -1 * torch.cos(theta_diff - estimated_theta_diff).mean()

class Model(nn.Module):
    def __init__(self, model_name = None, deploy = False,  bins=4, w = 0.4, input_size=(224, 224)):
        super(Model, self).__init__()
        self.bins = bins
        self.w = w
        self.deploy = deploy
        self.repvgg = repvgg.get_RepVGG_func_by_name(model_name)(deploy=self.deploy)
        self.repvgg = repvgg.repvgg_model_convert(self.repvgg)
        '''
        self.orientation = nn.Sequential(
                    nn.Linear(1000, 512),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(512, 256),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(256, bins*2) # to get sin and cos
                )
        '''
        self.confidence = nn.Sequential(
                    nn.Linear(1000, 512),
                    nn.ReLU(True),
                    #nn.Dropout(),
                    nn.Linear(512, 256),
                    nn.ReLU(True),
                    #nn.Dropout(),
                    nn.Linear(256, bins),
                    # nn.Softmax()
                    #nn.Sigmoid()
                )
        '''
        self.alpha = nn.Sequential(
                    nn.Linear(1000, 512),
                    nn.ReLU(True),
                    #nn.Dropout(),
                    nn.Linear(512, 256),
                    nn.ReLU(True),
                    #nn.Dropout(),
                    nn.Linear(256, 1) # to get sin and cos
                )
        '''
        '''self.dimension = nn.Sequential(
                    nn.Linear(1000, 512),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(512, 512),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(512, 3)
                )'''

    def forward(self, x):
        x = self.repvgg(x) # 1000
        x = x.view(-1, 1000)
        #orientation = self.orientation(x)
        #orientation = orientation.view(-1, self.bins, 2)
        #orientation = F.normalize(orientation, dim=2)
        confidence = self.confidence(x)
        #alpha = self.alpha(x)
        #dimension = self.dimension(x)
        return confidence
        #return alpha