# Doodle classification from scratch (XNDL, final lab)

Final graded lab of *Neural Networks and Deep Learning* (XNDL), Bachelor's Degree in Artificial Intelligence, FIB-UPC (2025-26).

The task: classify 32×32 grayscale doodles into 14 classes (apple, basketball, brain, circle, clock, compass, cookie, donut, face, moon, potato, sun, watermelon, wheel). Rules:

- No pretrained weights: the network is trained from scratch.
- Training runs on the course GPU (RTX 3080 or similar) and must finish in under 5 minutes.
- One single `.py` file, with the design decisions explained in comments.
- The grade is proportional to test accuracy, from 70% (pass) up to the best result in the class.

## Result

| | |
|---|---|
| Validation accuracy (course GPU) | **87.37%** in 196 s (90.20% train) |
| Grade | **9.93 / 10** (best result in the class: 88%) |

## Final model (`model_final.py`)

A ResNet9 adapted to small grayscale images:

- **Squeeze-and-excitation** blocks after each residual stage, so the network can weight channels by how useful they are for the current image.
- **Global average and max pooling concatenated** before the classifier. Max pooling catches a single strong feature (a hole in a donut); average pooling catches patterns spread over the whole drawing (the dots on a cookie or a potato).
- **Dropout** (0.25) before the final linear layer.

Training:

- **Data augmentation on the GPU**: horizontal flip, rotation, translation and cutout. Doing it on the GPU removed the CPU bottleneck, which mattered with a 5-minute budget. Cutout was added after looking at the confusion matrices: several classes share the same outline and differ only in a small detail, and hiding random patches stops the model from relying on a single feature.
- **SGD with Nesterov momentum and OneCycleLR**, plus label smoothing (0.1), weight decay and gradient clipping.
- **Mixed precision** (AMP) to fit more epochs into the budget.
- **Time-aware schedule**: the script times the first epoch and sets the number of remaining epochs so that the OneCycle schedule ends right when the time runs out. This keeps it working on GPUs of different speeds.
- **Test-time augmentation**: at evaluation, the prediction is averaged with that of the horizontally flipped image.

## Experiments

Validation accuracy of the main things tried. Most runs were done on Colab, which is slower than the grading GPU, so these numbers are a bit lower than the final one.

| File | Idea | Val. acc. |
|---|---|---|
| `Laboratori_XNDL_final_notat/model/simple.py` | Baseline CNN | ~80% |
| `experiments/preactresnet18.py` | PreActResNet18 | ~85% |
| `Laboratori_XNDL_final_notat/model/mobile_netv2.py` | MobileNetV2 | ~83% |
| `experiments/resnet9_mixup.py` | ResNet9 + mixup | 86.4% |
| `experiments/resnet9_cbam.py` | ResNet9 + CBAM attention (channel and spatial) | 87.0% |
| `experiments/two_stream_net.py` | Two branches: one for the global outline, one for local details | ~86% |
| `experiments/ensemble_cbam.py` | Ensemble of ResNet9+SE and ResNet9+CBAM | ~87% |
| `model_final.py` | ResNet9 + SE + avg/max pooling + TTA | **87.37%** |

Values marked with ~ are approximate: they were not logged in the files.

What I took from it:

- Going from a plain CNN to residual networks gave the biggest jump. After that, each change added a few tenths at most.
- The remaining errors are almost all between classes with the same round outline (circle, moon, basketball, wheel, potato) or with similar textures (cookie, potato, watermelon). The two-stream model and the attention variants were attempts to target exactly that.
- With a fixed time budget, every extra component has a cost in epochs, so the final choice was the simplest model among the best ones.

The folder `Laboratori_XNDL_final_notat/` contains the first iteration, built on the skeleton given by the course.

## Running it

The dataset is course material and is not included. Put `train.npz` and `val.npz` in `dades/dataset_npz/` at the repository root, then run from `practica2/`:

```bash
python model_final.py
```

Requires PyTorch with CUDA and torchvision.

Lab done in pairs (Laia Beyloc and Ilyas Ouacham).
