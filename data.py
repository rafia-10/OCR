from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from amharic_mapping import amharic_mapping

def preprocess_image(img_path):
    img = Image.open(img_path).convert('L')
    aspect_ratio = img.width / img.height
    new_width = int(32 * aspect_ratio)
    img = img.resize((new_width, 32), Image.BICUBIC)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    return transform(img)

def encode_label(label_text):
    encoded = []
    for char in label_text:
        for k,v in amharic_mapping.items():
            if v == char:
                encoded.append(k)
                break
        else:
            encoded.append(len(amharic_mapping))  # unknown
    return encoded

class OCRDataset(Dataset):
    def __init__(self, img_dir, labels_dir):
        self.img_dir = img_dir
        self.samples = []

        for fname in os.listdir(labels_dir):
            if fname.endswith('.txt'):
                with open(os.path.join(labels_dir, fname), 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 2: continue
                        img_name = parts[0]
                        label = ' '.join(parts[1:])
                        self.samples.append((img_name, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, label_text = self.samples[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img_tensor = preprocess_image(img_path)
        label_encoded = torch.tensor(encode_label(label_text), dtype=torch.long)
        return img_tensor, label_encoded

def collate_fn(batch):
    imgs, labels = zip(*batch)
    imgs = torch.stack(imgs, 0)
    label_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    labels_concat = torch.cat(labels)
    return imgs, labels_concat, label_lengths

def get_loaders(train_img_dir, train_labels_dir, val_img_dir, val_labels_dir, batch_size=8):
    from torch.utils.data import DataLoader
    train_dataset = OCRDataset(train_img_dir, train_labels_dir)
    val_dataset = OCRDataset(val_img_dir, val_labels_dir)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return train_loader, val_loader

