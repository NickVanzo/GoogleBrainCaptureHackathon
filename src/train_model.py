import os
import sys
import numpy as np
from copy import deepcopy
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import torch
from torch import nn
from sklearn.decomposition import PCA

sys.path[0] = os.getcwd()

from model.model import BendrEncoder
from model.model import Flatten
from omegaconf import OmegaConf

from src.data.conf.eeg_annotations import braincapture_annotations

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from torch.utils.data import DataLoader, TensorDataset

import logging

def main():
    # Suppress logger messages from MNE-Python
    mne_logger = logging.getLogger('training_log')
    mne_logger.setLevel(logging.INFO)
    mne_logger.info("Start training.") 
    
    fh = logging.FileHandler(os.path.join(os.getcwd(), 'log', 'spam.log'))
    fh.setLevel(logging.INFO)
    mne_logger.addHandler(fh)

    # set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model
    encoder = BendrEncoder()
    
    # Load configuration
    config = OmegaConf.load("src/parameter.yaml")["hyperparameters"]

    # Load the pretrained model
    encoder.load_state_dict(deepcopy(torch.load("encoder.pt", map_location=device)))
    encoder = encoder.to(device)

    # Load the datasets
    X = torch.load(os.path.join(config["data_path"], "X_data.pt")).to(device)
    y = torch.load(os.path.join(config["data_path"], "y_data.pt")).to(device).long()

    # Define model and configuration
    out_features = config.out_features

    linear_head = nn.Sequential(
        encoder,
        Flatten(),
        nn.Linear(in_features = 3 * 512 * 4, out_features = config.Linear_features, bias=True),
        nn.Dropout(p=0.4, inplace=False),
        nn.ReLU(),
        nn.BatchNorm1d(config.Linear_features),
        nn.Linear(config.Linear_features, out_features, bias=True) 
    )

    linear_head = linear_head.to(device)
    linear_head = linear_head.train()
    
    dataset = TensorDataset(X, y)

    batch_size = 4
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    learning_rate = config.learning_rate
    n_epochs = config.epochs

    optimizer = torch.optim.SGD(
        linear_head.parameters(), 
        lr=config.learning_rate, 
        weight_decay=config.weight_decay, 
        momentum=0.9, 
        nesterov=True)
    
    criterion = nn.CrossEntropyLoss()
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=learning_rate, 
        epochs=n_epochs, 
        steps_per_epoch=len(train_loader), 
        pct_start=0.1, 
        last_epoch=-1
    )
    
    finetune_model(linear_head, train_loader, criterion, optimizer, scheduler, n_epochs, device, mne_logger)
    mne_logger.info("End training.") 
    
    latent_representations = generate_latent_representations(X, encoder)
    mne_logger.info("Eval:") 
    evaluate_model(linear_head, test_loader, device, optimizer, mne_logger)
     
    
def generate_latent_representations(data, encoder, batch_size=5):
    """ Generate latent representations for the given data using the given encoder.
    Args:
        data (np.ndarray): The data to be encoded.
        encoder (nn.Module): The encoder to be used.
        batch_size (int): The batch size to be used.
    Returns:
        np.ndarray: The latent representations of the given data.
    """
    latent_size = (1536, 4) # do not change this 
    latent = np.empty((data.shape[0], *latent_size))

    for i in tqdm(range(0, data.shape[0], batch_size)):
        latent[i:i+batch_size] = encoder(data[i:i+batch_size]).cpu().detach().numpy()

    return latent.reshape((latent.shape[0], -1))
    
    
def finetune_model(model, train_loader, criterion, optimizer, scheduler, n_epochs, device, logger):

    for epoch in range(1, n_epochs + 1):
        total = correct = 0
        pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch}, train")
        for batch in train_loader:
            if len(batch[0]) < 2: continue            
            
            optimizer.zero_grad()
                    
            X, y = batch
            X, y = X.to(device), y.to(device)
            logits = model(X)       
            _, predicted = torch.max(logits.data, 1)

            total += y.size(0)
            correct += (predicted == y).sum().item()
            
            loss = criterion(logits, y)
            loss.backward()

            optimizer.step()
            scheduler.step()
            
            pbar.update(1)

        train_accuracy = np.round(100 * correct / total, 2)
        logger.info(f"Epoch {epoch}, train accuracy: {train_accuracy}%")

        
def evaluate_model(model, test_loader, device, optimizer, logger):
    with torch.no_grad():
        model.eval()
        total = correct = 0
        pbar = tqdm(total=len(test_loader), desc=f"Testing...")
        for batch in test_loader:
            if len(batch[0]) < 2: continue            
            
            optimizer.zero_grad()
                    
            X, y = batch
            X, y = X.to(device), y.to(device)

            logits = model(X)
            _, predicted = torch.max(logits.data, 1)

            total += y.size(0)
            correct += (predicted == y).sum().item()
            pbar.update(1)
            
        logger.info(f"Test accuracy: {100 * correct / total:2f}%") 


if __name__ == "__main__":
    main()

