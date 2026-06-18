'''Train con PyTorch (Híbrido Repo Original + Optimizaciones)'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms
import numpy as np

import os
import argparse
import time

from model.preactres import PreActResNet18
from model.simple import SimpleCNN
from model.mobile_netv2 import MobileNetV2
from model.boada_net import BoadaNet
from utils import progress_bar

parser = argparse.ArgumentParser(description='PyTorch XNDL Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best validation accuracy
start_epoch = 0 
MAX_TIME_SECONDS = 600 
print("DEVICE:", device)
# ---------------------------------------------------------
# 1. DATASET DE ALTO RENDIMIENTO (.NPZ)
# ---------------------------------------------------------
class NpzImageDataset(torch.utils.data.Dataset):
    def __init__(self, npz_path, transform=None):
        data = np.load(npz_path)
        self.images = data["images"]
        self.labels = data["labels"]
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # 1. Cargar imagen y pasar a rango [0, 1]
        img = self.images[idx].astype(np.float32) / 255.0
        # 2. Convertir a tensor y añadir la dimensión del canal (1, 32, 32)
        img = torch.from_numpy(img).unsqueeze(0)
        
        # 3. Aplicar Data Augmentation si existe (Rotaciones, recortes...)
        if self.transform:
            img = self.transform(img)
            
        # 4. Normalización estándar (-0.5 / 0.5) recomendada por el profe
        img = (img - 0.5) / 0.5
        label = int(self.labels[idx])
        
        return img, label

print('==> Preparing data..')
# IMPORTANTE: Hemos quitado ToTensor() porque la clase NpzImageDataset ya lo hace
transform_train = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), value=0)
])

# Directorio de los datos (Cámbialo según estés en Colab o en Boada)
DATA_DIR = './dades/dataset_npz/'

trainset = NpzImageDataset(os.path.join(DATA_DIR, "train.npz"), transform=transform_train)
# Subimos el batch_size a 512 para aprovechar la velocidad del .npz
trainloader = torch.utils.data.DataLoader(trainset, batch_size=512, shuffle=True, num_workers=2, pin_memory=True)

valset = NpzImageDataset(os.path.join(DATA_DIR, "val.npz"), transform=None) # Val no lleva transform
valloader = torch.utils.data.DataLoader(valset, batch_size=512, shuffle=False, num_workers=2, pin_memory=True)

# ---------------------------------------------------------
# 2. MODELO
# ---------------------------------------------------------
print('==> Building model..')

#net = PreActResNet18()  --> por ahora la mejor
#net = SimpleCNN()
#net = MobileNetV2()
net = BoadaNet()  #--> aun no probada

net = net.to(device)
if device == 'cuda':
    cudnn.benchmark = True

if args.resume:
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

# ---------------------------------------------------------
# 3. OPTIMIZADOR Y AMP
# ---------------------------------------------------------

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Definimos cuántas épocas reales esperamos que haga la RTX 3080 en Boada
MAX_EPOCHS_BOADA = 15 

optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-2)

scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,  
    max_lr=args.lr * 5, 
    steps_per_epoch=len(trainloader), 
    epochs=MAX_EPOCHS_BOADA
)
# Inicializamos el Scaler para la Precisión Mixta (AMP)
scaler = torch.amp.GradScaler('cuda')

# ---------------------------------------------------------
# 4. BUCLES DE ENTRENAMIENTO Y TEST
# ---------------------------------------------------------
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        
        # AMP: Ejecución en precisión de 16 bits para ir al doble de velocidad
        with torch.amp.autocast('cuda'):
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        scheduler.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # Barra de progreso original (Cuidado en Colab, puede saturar la pantalla)
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            with torch.amp.autocast('cuda'):
                outputs = net(inputs)
                loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(valloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc

# ---------------------------------------------------------
# 5. EJECUCIÓN CON LÍMITE DE TIEMPO
# ---------------------------------------------------------
start_time = time.time()

for epoch in range(start_epoch, start_epoch + 30):
    train(epoch)
    test(epoch)
    scheduler.step()
    
    elapsed_time = time.time() - start_time
    if elapsed_time > MAX_TIME_SECONDS:
        print(f"\n[!] Límite de tiempo alcanzado: {elapsed_time/60:.2f} min. Finalizando de forma segura.")
        break