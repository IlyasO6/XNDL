# Laboratori XNDL final notat - classificació d'imatges 32x32 en 14 categories

import os, time, copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np

#  hiperparàmetres 
DATA_DIR    = "../dades/dataset_npz/"
BATCH_SIZE  = 512
EPOCHS      = 35     
LR          = 0.1  
TIME_LIMIT  = 15 * 60

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

#  funció per obtenir les dades npz, però modificada respecte la original per problemes de coll d'ampolla de cpu
class NpzDataset(torch.utils.data.Dataset):
    def __init__(self, npz_path):
        print(f"Pre-loading {os.path.basename(npz_path)} into RAM...")
        data = np.load(npz_path)
        
        images_np = data["images"].astype(np.float32) / 255.0
        
        self.images = torch.from_numpy(images_np).unsqueeze(1)
        self.labels = torch.from_numpy(data["labels"]).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

#  hem aplicat transformacions de rotació i traslació, molt petites perquè els objectes s'assemblen bastant entre si,
#  i podem córrer el risc de que en aplicar rotació, per exemple, un objecte s'assembli a un altre
tf_train = transforms.Compose([
    transforms.RandomRotation(12), 
    transforms.RandomAffine(degrees=0, translate=(0.09, 0.09)),
    transforms.Normalize((0.1793,), (0.3324,))
])

tf_val = transforms.Compose([
    transforms.Normalize((0.1793,), (0.3324,))
])

#  carreguem les dades
train_ds = NpzDataset(os.path.join(DATA_DIR, "train.npz"))
val_ds   = NpzDataset(os.path.join(DATA_DIR, "val.npz"))


train_images = train_ds.images.to(device)  # ~459MB, one-time transfer
train_labels = train_ds.labels.to(device)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

#  llista de classes
classes = ['apple', 'basketball', 'brain', 'circle', 'clock', 'compass', 'cookie', 
           'donut', 'face', 'moon', 'potato', 'sun', 'watermelon', 'wheel']
n_classes = len(classes)
print(f"Classes ({n_classes}): {classes}")
print(f"Train: {len(train_ds):,}  |  Val: {len(val_ds):,}\n")

#  arquitectura del model, Resnet9 basat en el diseny de David Page
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool=False):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class ResNet9(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True)
        self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128), nn.Dropout(0.1))
        self.conv3 = ConvBlock(128, 256, pool=True)
        self.conv4 = ConvBlock(256, 512, pool=True)
        self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))
        self.classifier = nn.Sequential(
            nn.MaxPool2d(4), 
            nn.Flatten(), 
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out

model = ResNet9(in_channels=1, num_classes=n_classes).to(device)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Paràmetres entrenables: {n_params:,}\n")

#  hem afegit label_smoothing per ajudar amb l'overfitting
criterion = nn.CrossEntropyLoss(label_smoothing=0.1) 

#  mentre que Adam i AdamW convergien més rapidament, després de diverses proves hem trobat que SGD amb Nesterov's momentum
#  donava millors resultats de generalització
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=0.00125, nesterov=True)

#  hem creat un scheduler OneCycleLR després de fer recerca per poder afegir estabilitat a la optimització,
#  així com més velocitat de convergencia i millor generalització
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, steps_per_epoch=len(train_loader), 
                                          epochs=EPOCHS, pct_start=0.4)

#  per fer servir FP16, que és més rapid i se sol fer servir per tasques com aquesta
scaler = torch.amp.GradScaler('cuda')

def evaluate(loader):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            imgs = tf_val(imgs)
            
            with torch.amp.autocast('cuda'):
                out = model(imgs)
                loss = criterion(out, labels)
                
            loss_sum += loss.item() * len(labels)
            correct  += (out.argmax(1) == labels).sum().item()
            total    += len(labels)
    return correct / total, loss_sum / total

print(f"{'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  {'Val Loss':>8}  {'Val Acc':>7}")
print("─" * 55)

best_val_acc = 0.0
best_model_wts = copy.deepcopy(model.state_dict())
t_start = time.time()

for epoch in range(1, EPOCHS + 1):
    if time.time() - t_start > TIME_LIMIT:
        print(f"Temps exhaurit abans de l'epoch {epoch}. Aturant.")
        break
    print("OPTIMIZER param groups:", optimizer.param_groups[0]['lr'] )
    model.train()
    correct, total, loss_sum = 0, 0, 0.0

    perm = torch.randperm(len(train_labels), device=device)
    for i in range(0, len(perm), BATCH_SIZE):
        idx = perm[i:i+BATCH_SIZE]
        imgs, labels = train_images[idx], train_labels[idx]
        
        #  la data augmentation la fem aqui fent servir la gpu, per evitar el que mencionavem abans: el coll d'ampolla de la cpu.
        #  no sabem si aquesta es la millor solució, però ens ha funcionat per millorar el problema del temps significativament
        imgs = tf_train(imgs)
        
        optimizer.zero_grad()
        
        #  forward pass
        with torch.amp.autocast('cuda'):
            out  = model(imgs)
            loss = criterion(out, labels)
            
        #  backward pass
        scaler.scale(loss).backward()
        
        #  aquest pas l'hem afegit perquè ens donava un error de càlcul perque a vegades els càlculs es tornaven NaN. 
        #  es un pegat per ara ja que no hem tingut temps d'arreglar el core problem, que probablement es exploding gradient
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= scale_before:
            scheduler.step()
        loss_sum += loss.item() * len(labels)
        correct  += (out.argmax(1) == labels).sum().item()
        total    += len(labels)

    if total == 0:
        break
        
    train_acc  = correct / total
    train_loss = loss_sum / total
    val_acc, val_loss = evaluate(val_loader)

    print(f"{epoch:>5}  {train_loss:>10.4f}  {train_acc:>8.2%}  "
          f"{val_loss:>8.4f}  {val_acc:>6.2%}")

    #  guardem la millor accuracy
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_wts = copy.deepcopy(model.state_dict())


print(f"\nMillor val accuracy: {best_val_acc:.2%}")
print(f"Temps total: {time.time()-t_start:.0f}s")



model.load_state_dict(best_model_wts)
from sklearn.metrics import confusion_matrix
import numpy as np

model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, labels in val_loader:
        imgs = tf_val(imgs.to(device))
        out = model(imgs)
        all_preds.append(out.argmax(1).cpu())
        all_labels.append(labels)
cm = confusion_matrix(torch.cat(all_labels), torch.cat(all_preds))

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig('cm.png')