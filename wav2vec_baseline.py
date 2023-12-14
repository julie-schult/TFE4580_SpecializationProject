from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor, Wav2Vec2Model
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torchaudio
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import time

from plot_utils import save_model
from dataset import CustomDataset, CustomDataset2, CustomDatasetSplitZeros, CustomDatasetSplitRepeat

class Wav2Vec2Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.feature_extractor = self.model.feature_extractor
        
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(p=0.5)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(127744, 1)
        self.bn1 = nn.BatchNorm1d(1) 
        self.relu = nn.ReLU()

    def forward(self, input_values):
        x = self.feature_extractor(input_values)
        x = self.pooling(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x

    
    def fit(self, train_dataloader, val_dataloader=None, max_epochs=10, lr=0.000001, clip_value=1.0):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        val_accuracy_old = 0
        train_loss_all = []
        validation_loss_all = []
        
        for epoch in range(max_epochs):
            print('--------------------------------------')
            print(f'EPOCH {epoch+1}')
            self.train()
            train_loss = 0.0
            
            for inputs, labels in train_dataloader:
                inputs = inputs.to(torch.float32)
                optimizer.zero_grad()
                outputs = self.forward(inputs)
                            
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
                optimizer.step()
                
                train_loss +=loss.item()
            train_avg_loss = train_loss / len(train_dataloader)
            print(f'Training loss {train_avg_loss:.4f}')
            train_loss_all.append(train_avg_loss)
                
            if val_dataloader is not None:
                self.eval()
                
                with torch.no_grad():
                    val_loss = 0.0
                    correct = 0
                    total = 0
                
                    for inputs, labels in val_dataloader:
                        inputs = inputs.to(torch.float32)
                        outputs = self.forward(inputs)
                        probabilities = torch.sigmoid(outputs)
                        loss = criterion(outputs, probabilities)
                        val_loss += loss.item()
                        total += labels.size(0)
                        predictions = (probabilities > 0.5).float()
                        correct += torch.sum(predictions == labels).item()
            val_avg_loss = val_loss / len(val_dataloader)
            print(f'Validation loss: {val_avg_loss}')
            validation_loss_all.append(val_avg_loss)
            val_accuracy = correct / total
            print(f'Validation accuracy {val_accuracy} ({correct} out of {total})')
            
            
            if val_accuracy >= val_accuracy_old:
                val_accuracy_old = val_accuracy
                save_model(name='Baseline', model=self, optimizer=optimizer, epoch=epoch, val=val_accuracy)
                
        return train_loss_all, validation_loss_all
                
    def test(self, test_dataloader, weight_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        checkpoint = torch.load(weight_path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.eval()
        
        correct = 0
        total = 0

        all_predictions = []
        all_labels = []
        all_audios = []
        all_probabilities = []
        
        for inputs, labels in test_dataloader:
            
            outputs = self.forward(inputs)
            probabilities = torch.sigmoid(outputs)
            predicted = (probabilities > 0.5).int()
                        
            total += labels.size(0)
            labels = labels.int()
            correct += torch.sum(predicted == labels).item()
            
            predicted = predicted.tolist()
            label = labels.tolist()
            audio = inputs.tolist()
            probability = probabilities.tolist()
            
            all_audios.append(audio)
            all_labels.append(label)
            all_predictions.append(predicted)
            all_probabilities.append(probability)
        
        test_accuracy = correct / total
        print(f'Test accuracy: {test_accuracy}')
        
        return all_audios, all_labels, all_predictions, all_probabilities