import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import os
import time
import random
from PIL import Image 


def main():
  
    if os.path.exists('hymenoptera_data'):
        data_dir = 'hymenoptera_data'
    else:
        print("Folder not found")
        
    if not os.path.exists(data_dir):
        print(f"Error: Directory '{data_dir}' not found.")
        return

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- DATA PREPROCESSING ---
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    print("Loading data...")
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'val']}
    
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                 shuffle=True, num_workers=0)
                   for x in ['train', 'val']}
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    print(f"Classes: {class_names}")

    # --- MODEL SETUP ---
    print("\nDownloading and setting up ResNet18...")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)

    # --- TRAINING LOOP ---
    print(f"\nStarting training for 5 epochs...")
    
    for epoch in range(5):
        print(f'Epoch {epoch+1}/5')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    # --- TESTING ---
    print("\n--- Testing on Random Validation Images ---")
    val_samples = image_datasets['val'].samples
    random_picks = random.sample(val_samples, 3)

    for image_path, label_idx in random_picks:
        true_class = class_names[label_idx]
        print(f"\nImage: {image_path}")
        print(f"True Class: {true_class}")
        predict_custom_image(model, image_path, device, class_names)


def predict_custom_image(model, image_path, device, class_names):
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    try:
        image = Image.open(image_path)
        image = image.convert("RGB")
        image = transform(image).unsqueeze(0)
        image = image.to(device)

        model.eval()
        with torch.no_grad():
            outputs = model(image)
            _, preds = torch.max(outputs, 1)
            
        print(f"Prediction: {class_names[preds[0]]}")
    
    except Exception as e:
        print(f"Could not process image: {e}")


if __name__ == '__main__':
    main()