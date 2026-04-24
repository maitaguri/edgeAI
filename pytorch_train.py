import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os
import argparse

class ShirtDataset(Dataset):
    def __init__(self, csv_file, base_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        # NNC uses x:image and y:label headers
        self.img_paths = self.data['x:image'].values
        self.labels = self.data['y:label'].values
        self.base_dir = base_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        img_path = os.path.join(self.base_dir, self.img_paths[idx])
        img = Image.open(img_path).convert('L')
        img = img.resize((128, 128), Image.LANCZOS)
        
        # Convert to [0,1] tensor (1, H, W)
        import numpy as np
        img_np = np.array(img, dtype=np.float32) / 255.0
        img_np = np.expand_dims(img_np, axis=0) # add channel dim
        
        return torch.tensor(img_np), torch.tensor(self.labels[idx], dtype=torch.long)

class ShirtNet(nn.Module):
    def __init__(self):
        super(ShirtNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(64, 2)
        
    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x) # Logits
        return x

def main():
    print("Loading data...")
    dataset_dir = '/Users/maitaguri/Documents/B3/Experiments/EdgeAI/dataset'
    
    train_dataset = ShirtDataset(os.path.join(dataset_dir, 'train.csv'), dataset_dir)
    val_dataset = ShirtDataset(os.path.join(dataset_dir, 'val.csv'), dataset_dir)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    model = ShirtNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 15
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_acc = 0
        val_loss = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                val_acc += predicted.eq(labels).sum().item()
                
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss/len(train_loader):.4f} | Val Acc: {100.*val_acc/total:.2f}%")
        
    print("Exporting ONNX...")
    model.eval()
    dummy_input = torch.randn(1, 1, 128, 128)
    torch.onnx.export(model, dummy_input, "model.onnx", 
                      input_names=['x'], output_names=['y'],
                      opset_version=11)
    print("Exported model.onnx successfully!")

if __name__ == '__main__':
    main()
