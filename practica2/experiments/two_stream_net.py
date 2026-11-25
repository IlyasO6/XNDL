# Disseny des de zero — 32x32 grayscale, 14 classes de "doodles".
#
# El conjunt de classes té dos tipus d'error molt diferents (vist en les matrius
# de confusió d'iteracions anteriors):
#   1) confusió de SILUETA global: circle / potato / moon / basketball / wheel
#      es confonen perquè totes són "una taca rodona" — només es distingeixen
#      a escala grollera, no en detall fi.
#   2) confusió de DETALL local: cookie / potato / watermelon depenen d'ON hi ha
#      un tret (un forat, una taca, una agulla), no només de si hi és.
#
# En lloc d'una sola cadena de capes + atenció (que ha de servir per a tots dos
# problemes alhora), separem el model en dos camins explícits:
#   - ShapePath:  pocs canals, reducció agressiva i ràpida → veu la silueta.
#   - DetailPath: més capes a resolució alta, NO col·lapsa a 1x1 al final
#                 (manté una graella 2x2) → conserva una mica d'informació
#                 de posició fins al classificador.
#
# La part d'infraestructura (dades a GPU, AMP, clipping, probe automàtic de
# temps, OneCycleLR, TTA dins de forward()) es manté igual que ja teníem
# validada — això no és "arquitectura", són mecàniques d'entrenament ja
# depurades en iteracions anteriors.

import os, time, copy, math
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import numpy as np

# ── hiperparàmetres ──────────────────────────────────────────────────────────
DATA_DIR       = "../dades/dataset_npz/"
BATCH_SIZE     = 512
LR             = 0.4
TIME_LIMIT     = 25 * 60     # <- límit REAL d'avaluació (3080). NO és una aproximació de dev.
CUTOUT_SIZE    = 7
CUTOUT_PROB    = 0.3
USE_FOCAL_LOSS = True        # canvia a True per provar focal loss (definida més avall)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

classes = ['apple', 'basketball', 'brain', 'circle', 'clock', 'compass', 'cookie',
           'donut', 'face', 'moon', 'potato', 'sun', 'watermelon', 'wheel']
n_classes = len(classes)

# ── dades (precarregades a RAM i després a GPU, com ja teníem validat) ──────
class NpzDataset:
    def __init__(self, npz_path):
        print(f"Pre-loading {os.path.basename(npz_path)} into RAM...")
        data = np.load(npz_path)
        images_np = data["images"].astype(np.float32) / 255.0
        self.images = torch.from_numpy(images_np).unsqueeze(1)
        self.labels = torch.from_numpy(data["labels"]).long()
    def __len__(self):
        return len(self.labels)

train_ds = NpzDataset(os.path.join(DATA_DIR, "train.npz"))
val_ds   = NpzDataset(os.path.join(DATA_DIR, "val.npz"))

train_images = train_ds.images.to(device)
train_labels = train_ds.labels.to(device)
val_images   = val_ds.images.to(device)
val_labels   = val_ds.labels.to(device)

print(f"Train: {len(train_ds):,}  |  Val: {len(val_ds):,}\n")

# ── augmentació (GPU, batched) ───────────────────────────────────────────────
# cutout DESPRÉS de normalitzar (0.0 normalitzat == mitjana real, no negre cru —
# lliçó apresa en iteracions anteriors: omplir abans de normalitzar deixa un
# valor -0.54 que el model pot acabar usant com a senyal artificial).
def cutout(imgs, size=CUTOUT_SIZE, p=CUTOUT_PROB, n_holes=1):
    B, _, H, W = imgs.shape
    dev = imgs.device
    for _ in range(n_holes):
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
        mask = (in_box & apply.view(B, 1, 1)).unsqueeze(1)
        imgs = imgs.masked_fill(mask, 0.0)
    return imgs

tf_train = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),    # doodles sense lateralitat intrínseca
    transforms.RandomRotation(12),
    transforms.RandomAffine(degrees=0, translate=(0.09, 0.09)),
    transforms.Normalize((0.1793,), (0.3324,)),
    #cutout,
])
tf_val = transforms.Compose([
    transforms.Normalize((0.1793,), (0.3324,))
])

# ── arquitectura: dos camins explícits ──────────────────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, stride=stride, padding=pad, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)

class SE(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.shape
        s = self.fc(self.pool(x).view(b, c)).view(b, c, 1, 1)
        return x * s

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.b1 = ConvBlock(channels, channels)
        self.b2 = ConvBlock(channels, channels)
        self.se = SE(channels)
    def forward(self, x):
        out = self.b2(self.b1(x))
        return self.se(out + x)

class ShapePath(nn.Module):
    """Mirada ràpida i global a la silueta — per a circle/potato/moon/basketball/wheel."""
    def __init__(self, in_channels):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(in_channels, 48, stride=2, kernel_size=5),   # 32→16
            ConvBlock(48, 96, stride=2),                            # 16→8
            ConvBlock(96, 192, stride=2),                           # 8→4
            nn.AdaptiveAvgPool2d(1)
        )
    def forward(self, x):
        return self.net(x).flatten(1)   # (B, 128)

class DetailPath(nn.Module):
    """Manté resolució alta més temps i no col·lapsa a 1x1 — per a cookie/potato/watermelon."""
    def __init__(self, in_channels):
        super().__init__()
        self.stem  = nn.Sequential(ConvBlock(in_channels, 64), ConvBlock(64, 64))   # 32x32
        self.down1 = ConvBlock(64, 128, stride=2)    # 32→16 (stride en lloc de maxpool)
        self.res1  = ResBlock(128)
        self.down2 = ConvBlock(128, 256, stride=2)   # 16→8
        self.res2  = ResBlock(256)
        self.avg = nn.AdaptiveAvgPool2d(2)   # NO col·lapsem a 1x1 — graella 2x2
        self.max = nn.AdaptiveMaxPool2d(2)   # (quadrants: amunt/avall, esquerra/dreta)

    def forward(self, x):
        out = self.stem(x)
        out = self.down1(out)
        out = self.res1(out)
        out = self.down2(out)
        out = self.res2(out)
        avg_feat = self.avg(out).flatten(1)   # (B, 256*4)
        max_feat = self.max(out).flatten(1)   # (B, 256*4)
        return torch.cat([avg_feat, max_feat], dim=1)   # (B, 2048)

class TwoStreamNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.shape_path  = ShapePath(in_channels)
        self.detail_path = DetailPath(in_channels)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(192 + 2048, num_classes)
        self.use_tta = True

    def _forward_features(self, x):
        shape_feat  = self.shape_path(x)
        detail_feat = self.detail_path(x)
        feat = torch.cat([shape_feat, detail_feat], dim=1)
        return self.fc(self.dropout(feat))

    def forward(self, x):
        logits = self._forward_features(x)
        if self.training or not self.use_tta:
            return logits
        flipped = self._forward_features(torch.flip(x, dims=[3]))
        return (logits + flipped) / 2

model = TwoStreamNet(in_channels=1, num_classes=n_classes).to(device)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Paràmetres entrenables: {n_params:,}\n")

# ── loss ──────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """Alternativa a CE: redueix el pes dels exemples ja ben classificats
    (apple, sun, donut...) i concentra el gradient en els difícils
    (potato, moon, cookie...). Activa USE_FOCAL_LOSS=True més amunt per provar-ho."""
    def __init__(self, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none', label_smoothing=label_smoothing)
    def forward(self, logits, target):
        ce_loss = self.ce(logits, target)
        pt = torch.exp(-ce_loss)
        return (((1 - pt) ** self.gamma) * ce_loss).mean()

criterion = FocalLoss() if USE_FOCAL_LOSS else nn.CrossEntropyLoss(label_smoothing=0.1)

optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=0.0005, nesterov=True)
steps_per_epoch = math.ceil(len(train_ds) / BATCH_SIZE)
scheduler = None
scaler = torch.amp.GradScaler('cuda')

def evaluate():
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for i in range(0, len(val_labels), BATCH_SIZE):
            imgs   = tf_val(val_images[i:i + BATCH_SIZE])
            labels = val_labels[i:i + BATCH_SIZE]
            with torch.amp.autocast('cuda'):
                out  = model(imgs)
                loss = criterion(out, labels)
            loss_sum += loss.item() * len(labels)
            correct  += (out.argmax(1) == labels).sum().item()
            total    += len(labels)
    return correct / total, loss_sum / total

print(f"{'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  {'Val Loss':>8}  {'Val Acc':>7}")
print("─" * 55)

EPOCHS = 200   # placeholder — l'auto-probe el recalcula just després de l'epoch 1
best_val_acc = 0.0
best_model_wts = copy.deepcopy(model.state_dict())
initial_wts = copy.deepcopy(model.state_dict())
t_start = time.time()

epoch = 0
while epoch < EPOCHS:
    epoch += 1
    if time.time() - t_start > TIME_LIMIT:
        print(f"Temps exhaurit abans de l'epoch {epoch}. Aturant.")
        break

    if epoch == 1:
        for pg in optimizer.param_groups:
            pg['lr'] = LR * 0.01   # LR baix durant l'epoch de prova

    print("LR:", optimizer.param_groups[0]['lr'])
    model.train()
    correct, total, loss_sum = 0, 0, 0.0
    perm = torch.randperm(len(train_labels), device=device)

    for i in range(0, len(perm), BATCH_SIZE):
        if time.time() - t_start > TIME_LIMIT:
            print(f"Temps exhaurit durant l'epoch {epoch}. Aturant.")
            break
        idx = perm[i:i + BATCH_SIZE]
        imgs, labels = train_images[idx], train_labels[idx]
        imgs = tf_train(imgs)

        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            out  = model(imgs)
            loss = criterion(out, labels)

        scaler.scale(loss).backward()
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
    val_acc, val_loss = evaluate()

    if epoch == 1 and scheduler is None:
        first_epoch_time = time.time() - t_start
        remaining_budget = TIME_LIMIT - first_epoch_time
        estimated_remaining = max(10, int(remaining_budget / first_epoch_time * 0.97))
        EPOCHS = 1 + estimated_remaining
        print(f"  → Probe epoch: {first_epoch_time:.1f}s. Auto EPOCHS = {EPOCHS} "
              f"({estimated_remaining} remaining, LR schedule will complete)")
        model.load_state_dict(initial_wts)
        optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=0.0005, nesterov=True)
        scaler = torch.amp.GradScaler('cuda')
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=LR, steps_per_epoch=steps_per_epoch,
            epochs=estimated_remaining, pct_start=0.3
        )
        continue

    print(f"{epoch:>5}  {train_loss:>10.4f}  {train_acc:>8.2%}  "
          f"{val_loss:>8.4f}  {val_acc:>6.2%}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_wts = copy.deepcopy(model.state_dict())

print(f"\nMillor val accuracy: {best_val_acc:.2%}")
print(f"Temps total: {time.time() - t_start:.0f}s")

model.load_state_dict(best_model_wts)
torch.save(model.state_dict(), 'best_model.pt')

# ── matriu de confusió ──────────────────────────────────────────────────────
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for i in range(0, len(val_labels), BATCH_SIZE):
        imgs = tf_val(val_images[i:i + BATCH_SIZE])
        out  = model(imgs)
        all_preds.append(out.argmax(1).cpu())
        all_labels.append(val_labels[i:i + BATCH_SIZE].cpu())

cm = confusion_matrix(torch.cat(all_labels), torch.cat(all_preds))
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
plt.xlabel("Predicted"); plt.ylabel("True")
plt.savefig('cm_fresh2.png')