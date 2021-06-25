# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 14:45:49 2021

@author: vivek.muraleedharan
"""

import torch
import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self,input_size,hidden_state,num_classes):
        super(NeuralNet,self).__init__()
        self.l1= nn.Linear(input_size, hidden_state)
        self.l2= nn.Linear(hidden_state, hidden_state)
        self.l3= nn.Linear(hidden_state, num_classes)
        self.relu = nn.ReLU()
        
        
    def forward(self,x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        #no ativation and softmax
        return out
        