from super_gradients import init_trainer

from config import Config
import torch
import matplotlib.pyplot as plt
import os
import math
import random
from imutils import paths
from PIL import Image
from pathlib import Path, PurePath
from torchvision import transforms
import pandas as pd
from super_gradients.common.object_names import Models
from super_gradients.training import Trainer, models
from super_gradients.training.metrics.classification_metrics import Accuracy, Top5
from super_gradients.training.dataloaders.dataloaders import cifar10_train, cifar10_val
from super_gradients.training.utils.distributed_training_utils import setup_device
from typing import Dict, List,Tuple
import requests
import super_gradients
from super_gradients.common.object_names import Models
from super_gradients.training import Trainer
from super_gradients.training import training_hyperparams
from super_gradients.training.metrics.classification_metrics import Accuracy, Top5
from super_gradients.training.utils.early_stopping import EarlyStop
from super_gradients.training import models
from super_gradients.training.utils.callbacks import Phase
import torchvision
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
import numpy as np

def examine_images(images: list):
    num_images = len(images)
    num_rows = int(math.ceil(num_images / 3))
    num_cols = 5

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(30, 30), tight_layout=True)
    axs = axs.ravel()

    for i, image_path in enumerate(images[:num_images]):
        image = Image.open(image_path)
        label = PurePath(image_path).parent.name
        axs[i].imshow(image)
        axs[i].set_title(f"{label}", fontsize=20)
        axs[i].axis('off')
    plt.show()


def convert_l_to_rgb(image_path):
    img = Image.open(image_path)
    rgb_img = Image.merge('RGB', (img, img, img))
    rgb_img.save(image_path)


def create_dataloaders(
    train_dir: str,
    test_dir: str,
    valid_dir: str,
    batch_size: int,
    num_workers: int = 2
):
    train_data = datasets.ImageFolder(root=train_dir, transform=transforms.ToTensor())
    test_data = datasets.ImageFolder(root=test_dir, transform=transforms.ToTensor())
    valid_data = datasets.ImageFolder(root=valid_dir, transform=transforms.ToTensor())

    print(f"[INFO] training dataset contains {len(train_data)} samples...")
    print(f"[INFO] test dataset contains {len(test_data)} samples...")
    print(f"[INFO] validation dataset contains {len(valid_data)} samples...)")

    # Get class names
    names = train_data.classes
    print(f"[INFO] dataset contains {len(names)} labels...")

    # Turn images into data loaders
    print("[INFO] creating training and validation set dataloaders...")
    train_d_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    val_d_loader = DataLoader(
        valid_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True
    )

    test_d_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True
    )

    print(f"validation data: {valid_data}")
    print(f"validation loader dataset: {val_d_loader}")
    return train_d_loader, val_d_loader, test_d_loader, names


if __name__ == '__main__':
    train_image_path_list = list(sorted(paths.list_images(Config.TRAIN_DIR)))
    train_image_path_sample = random.sample(population=train_image_path_list, k=20)
    image_types = {'path': [], 'mode': []}
    for file_path in paths.list_images(Config.DATA_DIR):
        # open the image using PIL and get its mode
        with Image.open(file_path) as img:
            image_types['path'].append(file_path)
            image_types['mode'].append(img.mode)

    image_types_df = pd.DataFrame(image_types)

    for index, row in image_types_df.iterrows():
        if row['mode'] == 'L':
            convert_l_to_rgb(row['path'])
    # examine_images(train_image_path_sample)

    train_dataloader, valid_dataloader, test_dataloader, class_names = create_dataloaders(train_dir=Config.TRAIN_DIR,
                                                                                          test_dir=Config.TEST_DIR,
                                                                                          valid_dir=Config.VALID_DIR,
                                                                                          batch_size=Config.BATCH_SIZE)

    NUM_CLASSES = len(class_names)

    init_trainer()
    setup_device(device=Config.DEVICE)
    trainer = Trainer(experiment_name="Kaggle", ckpt_root_dir=Config.CHECKPOINT_DIR)
    model = models.get(Models.RESNET18, num_classes=4)
    training_params = {
        "max_epochs": 1,
        "initial_lr": 0.01,
        "loss": "cross_entropy",
        "train_metrics_list": [Accuracy()],
        "valid_metrics_list": [Accuracy()],
        "metric_to_watch": "Accuracy",
        "greate_metric_to_watch_is_better": True,
        "drop_last": True
    }

    trainer.train(model=model,
                  training_params=training_params,
                  train_loader=train_dataloader,
                  valid_loader=valid_dataloader)
    
    test_metrics = [Accuracy()]
    test_results = trainer.test(model=model, test_loader=test_dataloader, test_metrics_list=test_metrics)
    print(f"Test results: Accuracy: {test_results['Accuracy']}")

    # Set model to evaluation mode
    model.eval()

    # Create empty lists to store true labels and predicted labels
    true_labels = []
    predicted_labels = []

    # Loop over batches in test dataloader, make predictions, and append true and predicted labels to lists
    for images, labels in test_dataloader:
        images = images.to(Config.DEVICE)
        labels = labels.to(Config.DEVICE)
        with torch.no_grad():
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
        true_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(predicted.cpu().numpy())

    # Calculate confusion matrix, precision, and recall
    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    # Create figure and axis objects with larger size and font size
    fig, ax = plt.subplots(figsize=(12, 10))
    plt.rcParams.update({'font.size': 16})

    # Create heatmap of confusion matrix
    im = ax.imshow(conf_matrix, cmap='Blues')

    # Add colorbar to heatmap
    cbar = ax.figure.colorbar(im, ax=ax)

    # Set tick labels and axis labels with larger font size
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, fontsize=14)
    ax.set_yticklabels(class_names, fontsize=14)
    ax.set_xlabel('Predicted label', fontsize=16)
    ax.set_ylabel('True label', fontsize=16)

    # Rotate tick labels and set alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations to heatmap
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if conf_matrix[i, j] >= -1:  # Modify threshold value as needed
                text = ax.text(j, i, conf_matrix[i, j],
                            ha="center", va="center", color="y", fontsize=16)
            else:
                text = ax.text(j, i, "",
                            ha="center", va="center", color="y")

    # Add title to plot with larger font size
    ax.set_title("Confusion matrix", fontsize=20)

    # Show plot
    plt.show()