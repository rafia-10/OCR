
import torch
import torch.optim as optim
import torch.nn as nn
from data import get_loaders
from model import CRNN
from amharic_mapping import AMHARIC_MAPPING
import os


# Paths
BASE_DIR = '/content/drive/MyDrive/Cleaned_Data/Formatted Data'
TRAIN_IMG_DIR = os.path.join(BASE_DIR, 'train', 'cropped_images')
TRAIN_LABELS_DIR = os.path.join(BASE_DIR, 'train', 'labels')
VAL_IMG_DIR = os.path.join(BASE_DIR, 'val', 'cropped_images')
VAL_LABELS_DIR = os.path.join(BASE_DIR, 'val', 'labels')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NCLASS = len(AMHARIC_MAPPING) + 1  # +1 for CTC blank
BATCH_SIZE = 8

train_loader, val_loader = get_loaders(TRAIN_IMG_DIR, TRAIN_LABELS_DIR, VAL_IMG_DIR, VAL_LABELS_DIR, BATCH_SIZE)

model = CRNN(imgH=32, nc=1, nclass=NCLASS, nh=256).to(DEVICE)
criterion = nn.CTCLoss(blank=NCLASS-1)
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    model.train()
    total_loss = 0
    for imgs, labels, label_lengths in train_loader:
        print(f"Batch size: {imgs.size(0)}, Image shape: {imgs.shape}, Labels length: {label_lengths}")
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs).log_softmax(2)
        input_lengths = torch.full((imgs.size(0),), outputs.size(1), dtype=torch.long)
        loss = criterion(outputs.permute(1,0,2), labels, input_lengths, label_lengths)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} | Train Loss: {total_loss/len(train_loader):.4f}")


