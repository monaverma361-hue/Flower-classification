#%%
import torch
from torch import nn, optim
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np
import matplotlib.pyplot as plt
import os
import json

#%%
# -----------------------------
# Device setup
# -----------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Device set to:", device)

#%%
# -----------------------------
# Paths and parameters
# -----------------------------
data_dir = 'flower_data'
train_dir = os.path.join(data_dir, 'train')
valid_dir = os.path.join(data_dir, 'valid')
test_dir = os.path.join(data_dir, 'test')



#%%
# -----------------------------
# Data transforms
# -----------------------------
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

#%%
# -----------------------------
# Datasets & Dataloaders
# -----------------------------

batch_size = 32

datasets = {
    'train': ImageFolder(train_dir, transform=data_transforms['train']),
    'valid': ImageFolder(valid_dir, transform=data_transforms['valid']),
    'test': ImageFolder(test_dir, transform=data_transforms['test'])
}

dataloaders = {
    'train': DataLoader(datasets['train'], batch_size=batch_size, shuffle=True),
    'valid': DataLoader(datasets['valid'], batch_size=batch_size, shuffle=False),
    'test': DataLoader(datasets['test'], batch_size=batch_size, shuffle=False)
}



#%%
    # %%
def imshow(images, labels, idx_to_class, cat_to_name, n=4):
        """Display a batch of images with their flower names.

        Parameters:
        - images: batch of image tensors [B, C, H, W]
        - labels: tensor of class indices [B]
        - idx_to_class: dict mapping index -> class label (folder name)
        - cat_to_name: dict mapping class label -> flower name
        - n: number of images to display (default 4)"""

        images = images[:n]
        labels = labels[:n]

        fig, axes = plt.subplots(1, n, figsize= (4*n, 4))
        if n == 1:
            axes = [axes]  # Make it iterable

        for i,ax in enumerate(axes):
            img = images[i].numpy().transpose(1, 2, 0) # C,H,W to H,W,C
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = std * img + mean # unnormalize
            img = np.clip(img, 0, 1)

            class_labels = idx_to_class[labels[i].item()]
            flower_name = cat_to_name[class_labels]

            ax.imshow(img)
            ax.set_title(flower_name)
            ax.axis('off')

        plt.tight_layout()
        plt.show()

    # Reverse mapping
train_dataset = datasets['train']
idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}

    # Load your JSON (use path relative to this script)
cat_file = os.path.join(os.path.dirname(__file__), 'cat_to_name.json')
with open(cat_file, 'r') as f:
        cat_to_name = json.load(f)

    # Get one batch
images, labels = next(iter(dataloaders['train']))

    # Show first 6 images with flower names
imshow(images, labels, idx_to_class, cat_to_name, n=6)

#%%
# -----------------------------
# Model setup (ResNet50)
# -----------------------------
model = models.resnet50(pretrained=True)

# Freeze entire backbone
for param in model.parameters():
    param.requires_grad = False

# Replace the final fully connected layer
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 102)  # 102 flower classes
model = model.to(device)



# -----------------------------
# Loss, optimizer, scheduler
# -----------------------------


learning_rate = 0.0005
epochs_head = 5
epochs_finetune = 6
print_every = 5

#%%
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.fc.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0001)

scheduler = OneCycleLR(
    optimizer,
    max_lr=0.01,
    steps_per_epoch=len(dataloaders['train']),
    epochs=epochs_head,
    anneal_strategy='cos'
)

# -----------------------------
# Training loop
# -----------------------------


#%%
print("Starting training of the new head...")
train_losses_head, val_losses_head, val_acc_head = [], [], []
steps = 0

for epoch in range(epochs_head):
    running_loss = 0
    model.train()
    for inputs, labels in dataloaders['train']:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()

        # Validation
        if steps % print_every == 0:
            model.eval()
            val_loss = 0
            accuracy = 0
            with torch.no_grad():
                for val_inputs, val_labels in dataloaders['valid']:
                    val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                    val_outputs = model(val_inputs)
                    batch_loss = criterion(val_outputs, val_labels)
                    val_loss += batch_loss.item()

                    preds = val_outputs.argmax(dim=1)
                    accuracy += torch.sum(preds == val_labels).item() / len(val_labels)

            train_losses_head.append(running_loss / print_every)
            val_losses_head.append(val_loss / len(dataloaders['valid']))
            val_acc_head.append(accuracy / len(dataloaders['valid']))

            print(f"Epoch {epoch+1}/{epochs_head} | Step {steps} | "
                  f"Train Loss: {running_loss/print_every:.3f} | "
                  f"Val Loss: {val_loss/len(dataloaders['valid']):.3f} | "
                  f"Val Acc: {accuracy/len(dataloaders['valid']):.3f}")

            running_loss = 0
            model.train()
#%%
# save head only checkpoint
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epochs': epochs_head,
    'class_to_idx': datasets['train'].class_to_idx,
    'classifier': model.fc
}, 'checkpoint_resnet_head.pth')

#%%
# save head training metrics
train_losses = train_losses_head
val_losses = val_losses_head
val_accuracies = val_acc_head
# -----------------------------
# Plot metrics
# -----------------------------
os.makedirs('output', exist_ok=True)
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Step (per print_every)")
plt.ylabel("Loss")
plt.title("Training & Validation Loss")
plt.legend()

plt.subplot(1,2,2)
plt.plot(val_accuracies, label="Validation Accuracy", color='green')
plt.xlabel("Step (per print_every)")
plt.ylabel("Accuracy")
plt.title("Validation Accuracy")
plt.legend()

plt.savefig('output/training_validation_head_metrics.png', dpi=300, bbox_inches='tight')
plt.show()

#%%
# Finetuning the unfreeze last 2 layers
for name, param in model.named_parameters():
    if "layer4" in name or "layer3" in name:
        param.requires_grad = True
    
# New optimizer for finetuning
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001, momentum=0.9, weight_decay=0.0001)
scheduler = OneCycleLR(
    optimizer,
    max_lr=1e-4,
    steps_per_epoch=len(dataloaders['train']),
    epochs=epochs_finetune,
    anneal_strategy='cos'
)
print("Starting finetuning of last 2 layers...")
train_losses_ft, val_losses_ft, val_acc_ft = [], [], []
steps = 0
for epoch in range(epochs_finetune):
    running_loss = 0
    model.train()
    for inputs, labels in dataloaders['train']:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()

        # Validation
        if steps % print_every == 0:
            model.eval()
            val_loss = 0
            accuracy = 0
            with torch.no_grad():
                for val_inputs, val_labels in dataloaders['valid']:
                    val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                    val_outputs = model(val_inputs)
                    batch_loss = criterion(val_outputs, val_labels)
                    val_loss += batch_loss.item()

                    preds = val_outputs.argmax(dim=1)
                    accuracy += torch.sum(preds == val_labels).item() / len(val_labels)

            train_losses_ft.append(running_loss / print_every)
            val_losses_ft.append(val_loss / len(dataloaders['valid']))
            val_acc_ft.append(accuracy / len(dataloaders['valid']))

            print(f"Epoch {epoch+1}/{epochs_finetune} | Step {steps} | "
                  f"Train Loss: {running_loss/print_every:.3f} | "
                  f"Val Loss: {val_loss/len(dataloaders['valid']):.3f} | "
                  f"Val Acc: {accuracy/len(dataloaders['valid']):.3f}")

            running_loss = 0
            model.train()

#%%
# -----------------------------
# Save finetuned model checkpoint
# -----------------------------
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epochs': epochs_finetune,
    'class_to_idx': datasets['train'].class_to_idx,
    'classifier': model.fc
}, 'checkpoint_resnet_finetune.pth')

#%%
# plot finetuning metrics
train_losses = train_losses_ft
val_losses = val_losses_ft
val_accuracies = val_acc_ft

os.makedirs('output', exist_ok=True)
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Step (per print_every)")
plt.ylabel("Loss")
plt.title("Training & Validation Loss")
plt.legend()

plt.subplot(1,2,2)
plt.plot(val_accuracies, label="Validation Accuracy", color='green')
plt.xlabel("Step (per print_every)")
plt.ylabel("Accuracy")
plt.title("Validation Accuracy")
plt.legend()

plt.savefig('output/training_validation_finetune_metrics.png', dpi=300, bbox_inches='tight')
plt.show()



# -----------------------------
# Test evaluation
# -----------------------------
model.eval()
test_loss = 0
test_correct = 0
with torch.no_grad():
    for inputs, labels in dataloaders['test']:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        test_loss += criterion(outputs, labels).item()

        preds = outputs.argmax(dim=1)
        test_correct += torch.sum(preds == labels).item()

test_accuracy = test_correct / len(datasets['test'])
print(f"Test Loss: {test_loss/len(dataloaders['test']):.3f} | Test Accuracy: {test_accuracy:.3f}")