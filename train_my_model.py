import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from my_model import MyCustomCNN


import os
from PIL import Image
from torch.utils.data import Dataset

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.images = os.listdir(img_dir)

        # Aici etichetezi imaginile după un criteriu – exemplu: numele fișierului
        self.labels = []
        for img_name in self.images:
            if "shot" in img_name.lower():
                self.labels.append(1)
            else:
                self.labels.append(0)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# Setări
batch_size = 16
epochs = 10
lr = 0.001
img_size = 256  # trebuie să fie 64 x 64 pentru fully connected layer în modelul tău

# Transformări imagini
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor()
])

# Încarcă datele de antrenare
train_dataset = CustomImageDataset(img_dir='Dataset/train/images', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Inițializează modelul
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MyCustomCNN().to(device)

# Loss și optimizer
criterion = nn.BCELoss()  # binary cross entropy pentru ieșire cu sigmoid
optimizer = optim.Adam(model.parameters(), lr=lr)

# Antrenare
for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss:.4f}")

# Salvează modelul
torch.save(model.state_dict(), 'my_custom_model.pt')
print("Model salvat ca my_custom_model.pt")
