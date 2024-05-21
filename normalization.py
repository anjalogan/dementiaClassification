import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import torch
from pathlib import Path
from config import Config
from super_gradients import init_trainer
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

def make_data_loaders(batch_size, num_workers: int = 2):
    train_data = list_data("train")
    #print(len(train_data))
    test_data = list_data("test")
    #print(len(test_data))
    validate_data = list_data("validate")
    #print(len(validate_data))

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
        validate_data,
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

    return train_d_loader, val_d_loader, test_d_loader

def list_data(type):
    data = []
    destination_path = 'data/'+type
    format_of_your_images = 'jpg'
    path = Path(destination_path).rglob(f'*.{format_of_your_images}')
    #print(path)

    for filename in path: #assuming jpg
        img = cv2.imread(str(filename))
        class_name = str(filename).split("\\")[2]
        if class_name == "MildDemented": class_num = 0
        if class_name == "ModerateDemented": class_num = 1
        if class_name == "NonDemented": class_num = 2
        if class_name == "VeryMildDemented": class_num = 3
        img_n = cv2.normalize(img, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        tuple_data = (torch.from_numpy(img_n).permute(2,0,1),class_num)
        data.append(tuple_data)
    
    return data

# this function to read in images, normalize them all and then save them isn't working very well. When I look at the images I can't see anything
def normalize_save():
    destination_path = 'data/test/ModerateDemented'
    target_path = 'data_norm'

    format_of_your_images = 'jpg'

    all_the_files = Path(destination_path).rglob(f'*.{format_of_your_images}')

    for f in all_the_files:
        p = cv2.imread(str(f))
        p_norm = cv2.normalize(p, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        cv2.imwrite(f'{target_path}/{f.name}', p_norm)

def main():
    train_dataloader, valid_dataloader, test_dataloader = make_data_loaders(batch_size=Config.BATCH_SIZE)
    class_names = ["MildDemented", "ModerateDemented", "NonDemented", "VerMildDemented"]
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

if __name__ == '__main__':
    main()