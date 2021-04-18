import torch
from torch import nn

class GRUModel(nn.Module):
    def __init__(self, input_size=40, hidden_size=256, num_classes = 2):
        super().__init__()
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers=2)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.initHidden()
        self.inp_size = input_size

    def forward(self, x):
        x = x.view(-1, 1, self.inp_size)

        out, self.h1 = self.gru(x, self.h1)
        self.h1 = self.h1.detach()
        out = nn.Tanh()(out)
        out = self.fc(out)
        B, N, D = out.size()
        
        return out.view(-1, D)
    
    def initHidden(self):
        self.h1 = None
        self.h2 = None
