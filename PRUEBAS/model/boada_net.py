import torch
import torch.nn as nn

class BoadaNet(nn.Module):
    def __init__(self, num_classes=14):
        super().__init__()
        
        self.features = nn.Sequential(
            # Bloque 1: 32x32 -> 16x16
            nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Bloque 2: 16x16 -> 8x8
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Bloque 3: 8x8 -> 4x4
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Bloque 4: 4x4 -> 2x2
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            # Decisión Clave Pro: Global Average Pooling
            # Reduce cualquier dimensión espacial restante a 1x1
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Clasificador final súper ligero
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x