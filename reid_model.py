import torch
import torch.nn as nn
from frw import FRW

class ReID_Net(nn.Module):
    """ CNN used in the Paper 'Deep Person Re-Identification with Improved Embedding 
        and Efficient Training' (https://arxiv.org/abs/1705.03332).
        LeakyReLUs replaced by PReLUs, since the leak rate was not informed.
    """
    def __init__(self, num_classes, feature_dim=512):
        super().__init__()
        
        self.convolutions = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(96, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.Conv2d(128, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.fc_embeddings = nn.Sequential(
            nn.Linear(9216, 512),
            nn.BatchNorm2d(512),
            nn.PReLU(),
            FRW(512)
        )
        
        self.classifier = nn.Linear(512, num_classes)
        
        # Buffer for the `centers` tensor. It is updated by the CenterLoss class
        # during training.
        self.register_buffer('centers', torch.randn(num_classes, feature_dim))
        
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        
        
    def forward(self, x):
        x = self.convolutions(x)
        # Flatten
        x = x.view(x.size(0), -1)
        features = self.fc_embeddings(x)
        x = self.classifier(features)
        
        return features, F.log_softmax(x)
    
    
    def get_embeddings(self, x):
        x = self.convolutions(x)
        x = x.view(x.size(0), -1)
        features = self.fc_embeddings(x)
        return features
