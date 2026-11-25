# Laboratori XNDL final notat - classificació d'imatges 32x32 en 14 categories
#  GRUP: Laia Beyloc Guzman i Ilyas Ouacham Rhailane
#  mails: laia.beyloc@estudiantat.upc.edu / ilyas.ouacham@estudiantat.upc.edu
import os, time, copy, math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np

#  hiperparàmetres 
DATA_DIR = "../dades/dataset_npz"
BATCH_SIZE  = 512
EPOCHS      = 200   
LR          = 0.1  
TIME_LIMIT  = 10 * 60
CUTOUT_SIZE = 5    
CUTOUT_PROB = 0.5



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

#  funció per obtenir les dades npz, però modificada respecte la original per problemes de coll d'ampolla de cpu
class NpzDataset(torch.utils.data.Dataset):
    def __init__(self, npz_path):
        print(f"Carregant {os.path.basename(npz_path)} a la RAM...")
        data = np.load(npz_path)
        
        images_np = data["images"].astype(np.float32) / 255.0
        
        self.images = torch.from_numpy(images_np).unsqueeze(1)
        self.labels = torch.from_numpy(data["labels"]).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


#  Hem implementat cutout perquè, després d'analitzar les imatges manualment, hem trobat
#  que hi ha certes classes que es confonen facilment, i que tenen petits trets que les diferencien.
#  Així doncs, amb cutout evitem que el model aprengui la mateixa característica per a totes aquestes classes 
#  (les que es confonen), i intentem que aprengui trets més definidors. 
def cutout(imgs, size=CUTOUT_SIZE, p=CUTOUT_PROB, n_holes=1):  
    for _ in range(n_holes):
        B, _, H, W = imgs.shape
        dev = imgs.device
        apply = torch.rand(B, device=dev) < p
        half = size // 2
        cy = torch.randint(0, H, (B,), device=dev)
        cx = torch.randint(0, W, (B,), device=dev)
        y1, y2 = (cy - half).clamp(0, H), (cy + half).clamp(0, H)
        x1, x2 = (cx - half).clamp(0, W), (cx + half).clamp(0, W)
    
        y_idx = torch.arange(H, device=dev).view(1, H, 1)
        x_idx = torch.arange(W, device=dev).view(1, 1, W)
        in_box = ((y_idx >= y1.view(B, 1, 1)) & (y_idx < y2.view(B, 1, 1)) &
                (x_idx >= x1.view(B, 1, 1)) & (x_idx < x2.view(B, 1, 1)))
        mask = (in_box & apply.view(B, 1, 1)).unsqueeze(1)  # (B,1,H,W)
        imgs = imgs.masked_fill(mask, 0.0)  
    return imgs  


#  hem aplicat transformacions de rotació i traslació, molt petites perquè els objectes s'assemblen bastant entre si,
#  i podem córrer el risc de que en aplicar rotació, per exemple, un objecte s'assembli a un altre.
#  A més, hem aplicat el cutout després de la normalització, perquè la "taca" del cutout tingui valor 0 (normalitzada).
#  Finalment, cal aclarir que la mitjana i la desviació la hem calculat amb un altre programa a partir de les dades de train.
tf_train = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),  
    transforms.RandomRotation(15),  
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), 
    transforms.Normalize((0.1793,), (0.3324,)),
    cutout, 
])

tf_val = transforms.Compose([
    transforms.Normalize((0.1793,), (0.3324,))
])

#  carreguem les dades
train_ds = NpzDataset(os.path.join(DATA_DIR, "train.npz"))
val_ds   = NpzDataset(os.path.join(DATA_DIR, "val.npz"))


train_images = train_ds.images.to(device) 
train_labels = train_ds.labels.to(device)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

#  llista de classes
classes = ['apple', 'basketball', 'brain', 'circle', 'clock', 'compass', 'cookie', 
           'donut', 'face', 'moon', 'potato', 'sun', 'watermelon', 'wheel']
n_classes = len(classes)
print(f"Classes ({n_classes}): {classes}")
print(f"Train: {len(train_ds):,}  |  Val: {len(val_ds):,}\n")


#  Squeeze-and-excitation
#  Hem fet servir aquest tipus d'atenció en comptes d'atenció espacial perquè
#  hem fet proves i ha donat millors resultats i menys temps de computació. 
#  L'atenció l'hem implementat per intentar captar les petites diferencies entre 
#  les classes altamente confundides.
class SE(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.shape
        scale = self.squeeze(x).view(b, c)
        scale = self.excitation(scale).view(b, c, 1, 1)
        return x * scale


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
        self.conv1b = ConvBlock(64, 64) 
        self.conv2 = ConvBlock(64, 128, pool=True)       # 32→16
        self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))
        self.se1 = SE(128)  
        self.conv3 = ConvBlock(128, 256, pool=True)       # 16→8
        self.conv4 = ConvBlock(256, 512, pool=False)      
        self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))
        self.se2 = SE(512)  
 
        #  combinem avg-pool i max-pool globals: el max captura un tret puntual fort
        #  (p.ex. un forat), l'avg captura un patró distribuït per tota la superfície
        #  (p.ex. taques de cookie/potato), cosa que MaxPool2d sol no veu bé
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.dropout = nn.Dropout(0.25) 
        self.fc = nn.Linear(1024, num_classes)   # 512 (avg) + 512 (max)

        self.use_tta = True
        
    def _forward_features(self, x):
        out = self.conv1(x)
        out = self.conv1b(out)
        out = self.conv2(out)
        out = self.se1(self.res1(out) + out)  
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.se2(self.res2(out) + out)  
        avg_pooled = self.gap(out).flatten(1)
        max_pooled = self.gmp(out).flatten(1)
        pooled = torch.cat([avg_pooled, max_pooled], dim=1)
        return self.fc(self.dropout(pooled))

    def forward(self, x):
        logits = self._forward_features(x)
        if self.training or not self.use_tta:
            return logits                     
        logits_flipped = self._forward_features(torch.flip(x, dims=[3]))
        return (logits + logits_flipped) / 2   

model = ResNet9(in_channels=1, num_classes=n_classes).to(device)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Paràmetres entrenables: {n_params:,}\n")

#  hem afegit label_smoothing per ajudar amb l'overfitting
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

#  mentre que Adam i AdamW convergien més rapidament, després de diverses proves hem trobat que SGD amb Nesterov's momentum
#  donava millors resultats de generalització
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=0.00125, nesterov=True)


steps_per_epoch = math.ceil(len(train_ds) / BATCH_SIZE)

scheduler = None  

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
initial_wts = copy.deepcopy(model.state_dict())  
t_start = time.time()

epoch = 0
while epoch < EPOCHS: 
    epoch += 1
    epoch_start = time.time()

    if time.time() - t_start > TIME_LIMIT:
        print(f"Temps exhaurit abans de l'epoch {epoch}. Aturant.")
        break

    if epoch <= 2:
        for pg in optimizer.param_groups:
            pg['lr'] = LR * 0.01  

    print("OPTIMIZER param groups:", optimizer.param_groups[0]['lr'] )
    model.train()
    correct, total, loss_sum = 0, 0, 0.0

    perm = torch.randperm(len(train_labels), device=device)
    for i in range(0, len(perm), BATCH_SIZE):
        if time.time() - t_start > TIME_LIMIT:
            print(f"Temps exhaurit durant l'epoch {epoch}. Aturant.")
            break
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
        
        #  vam afegir gradient clipping perquè abans es donava el cas de exploding gradient
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()

        if scaler.get_scale() >= scale_before:
            if scheduler is not None and scheduler._step_count < scheduler.total_steps:  
                scheduler.step()
        loss_sum += loss.item() * len(labels)
        correct  += (out.argmax(1) == labels).sum().item()
        total    += len(labels)

    if total == 0:
        break
        
    train_acc  = correct / total
    train_loss = loss_sum / total

    val_acc, val_loss = evaluate(val_loader)

    #  tot aquest bloc el fem servir per calcular el nombre d'epochs exacte que executarem, de cara
    #  a poder configurar correctament el scheduler OneCycleLR, que depen d'aquest nombre
    if epoch == 2 and scheduler is None:
        first_epoch_time = time.time() - epoch_start
        remaining_budget = TIME_LIMIT - (time.time() - t_start)
        
        estimated_remaining = max(12, int(remaining_budget / first_epoch_time * 0.95))
        EPOCHS = 2 + estimated_remaining
        print(f"  --> Primer epoch: {first_epoch_time:.1f}s. EPOCHS = {EPOCHS} ")
        
        model.load_state_dict(initial_wts)
        optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=0.00125, nesterov=True)
        scaler = torch.amp.GradScaler('cuda')
        #  hem creat un scheduler OneCycleLR després de fer recerca per poder afegir estabilitat a la optimització,
        #  així com més velocitat de convergencia i millor generalització
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=LR,
            steps_per_epoch=steps_per_epoch,
            epochs=estimated_remaining,
            pct_start=0.3,
        )
        continue  

    print(f"{epoch:>5}  {train_loss:>10.4f}  {train_acc:>8.2%}  "
          f"{val_loss:>8.4f}  {val_acc:>6.2%}")

    #  guardem la millor accuracy
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pt')


print(f"\nMillor val accuracy: {best_val_acc:.2%}")
print(f"Temps total: {time.time()-t_start:.0f}s")

