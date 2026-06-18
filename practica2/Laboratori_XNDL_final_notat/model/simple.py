import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=14):
        super().__init__()
        
        # Extractor de características espaciales
        self.features = nn.Sequential(
            # BLOQUE 1: Detectar bordes simples (Imagen pasa de 32x32 a 16x16)
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), # Decisión Clave 2
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # Decisión Clave 3
            
            # BLOQUE 2: Detectar formas complejas (Imagen pasa de 16x16 a 8x8)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # BLOQUE 3: Conceptos abstractos (Imagen pasa de 8x8 a 4x4)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        
        # Clasificador Final
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3), # Regularización clásica para evitar memorizar
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x