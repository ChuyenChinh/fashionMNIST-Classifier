import torch 
import torch.nn as nn

class fashionMLP:
    def __init__(self):
        super().__init__()
        self.flat = nn.Flatten()
        self.l1 = nn.Linear(784,128)
        self.l2 = nn.ReLU()
        self.l3 = nn.Linear(128,64)
        self.l4 = nn.ReLU()
        self.l5 = nn.Linear(64,10)
    def forward(self,x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        return self.l5(x)