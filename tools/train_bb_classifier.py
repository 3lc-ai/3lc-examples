import os

import timm
import tlc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# Configuration settings
data_dir = "./data"

train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "val")
batch_size = 32
num_classes = 4
num_epochs = 20
learning_rate = 0.0001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_save_path = "trained_model.pth"

PROJECT_NAME = "DCVAI-Classification"

# Data preprocessing
_train_transform = transforms.Compose(
    [
        # transforms.Resize((224, 224)),  # Resize images to 224x224
        transforms.RandomHorizontalFlip(),  # Data augmentation
        transforms.RandomRotation(10),  # Random rotation
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # Normalization for ImageNet pre-trained models
    ]
)


def train_transform(sample):
    image, label = sample
    return (_train_transform(image), label)


_val_transform = transforms.Compose(
    [
        # transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def val_transform(sample):
    image, label = sample
    return (_val_transform(image), label)


# Load datasets
train_dataset = (
    tlc.Table.from_image_folder(
        root=train_dir, project_name=PROJECT_NAME, dataset_name="dcvai_train", table_name="initial"
    )
    .map(train_transform)
    .map_collect_metrics(val_transform)
    .revision(table_name="fix-weights")
)
val_dataset = (
    tlc.Table.from_image_folder(root=val_dir, project_name=PROJECT_NAME, dataset_name="dcvai_val", table_name="initial")
    .map(val_transform)
    .revision(table_name="fix-weights")
)

# sampler = train_dataset.create_sampler()
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4, sampler=sampler)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Initialize the model from timm
model = timm.create_model("resnet50", pretrained=True, num_classes=num_classes)
model = model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Add a learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.1)


# Function to train the model
def train_model(model, num_epochs=25):
    best_acc = 0.0
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
                dataloader = train_loader
            else:
                model.eval()  # Set model to evaluate mode
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in tqdm(dataloader, desc=f"{phase} epoch {epoch+1}/{num_epochs}"):
                inputs, labels = inputs.to(device), labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                # Track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            tlc.log(
                {
                    "epoch": epoch,
                    f"{phase}_loss": epoch_loss,
                    f"{phase}_acc": epoch_acc,
                    "LR": optimizer.param_groups[0]["lr"],
                }
            )

            print(f"{phase} loss: {epoch_loss:.4f} acc: {epoch_acc:.4f}")

            if phase == "val":
                if epoch_acc > best_acc:
                    # Deep copy the model if accuracy improves
                    best_acc = epoch_acc
                    best_model_wts = model.state_dict()

                # Step the scheduler based on validation loss
                curr_lr = scheduler.get_last_lr()[0]
                scheduler.step(epoch_loss)
                new_lr = scheduler.get_last_lr()[0]
                if curr_lr != new_lr:
                    print(f"LR changed from {curr_lr} to {new_lr}")

    print(f"Best val Acc: {best_acc:.4f}")

    # Load best model weights
    model.load_state_dict(best_model_wts)

    tlc.collect_metrics(
        train_dataset,
        tlc.ClassificationMetricsCollector(["Boy", "Girl", "Man", "Woman"]),
        model,
        split="train",
    )
    tlc.collect_metrics(
        val_dataset,
        tlc.ClassificationMetricsCollector(["Boy", "Girl", "Man", "Woman"]),
        model,
        split="val",
    )

    return model


if __name__ == "__main__":
    # Train and validate the model

    run = tlc.init(
        PROJECT_NAME, "train-classifier", "Train a classifier for the DCVAI dataset (better labels, no weights)"
    )

    trained_model = train_model(model, num_epochs=num_epochs)

    # Save the trained model
    torch.save(trained_model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    run.set_status_completed()
