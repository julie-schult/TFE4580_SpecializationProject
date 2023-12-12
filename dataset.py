from torch.utils.data import Dataset
import pandas as pd
import os.path
import torchaudio
import torch
import torch.nn as nn
from help_functions import train_val_test, split_audios_sec


# Big dataset that splits the files, and cut them into smaller segments if needed
def CustomDatasetSplitZeros(csv_path):
    
    # train val test splitting
    dfs = train_val_test(csv_path)

    # split waveforms if needed, and pad
    Xs = []
    ys = []
    ns = []
    for df in dfs:
        X = []
        y = []
        n = []
        for idx, row in df.iterrows():
            path = row['path']
            label = row['label']
    
            all_clips, all_names = split_audios_sec(path)
            if len(all_clips)==0:
                waveform, _ = torchaudio.load(path)
                # padding, so all lengths are 319925
                waveform_padded = torch.nn.functional.pad(waveform, (0, 319925 - waveform.size(1)))
                waveform_padded = waveform_padded.squeeze(0)
                X.append(waveform_padded)
                y.append(label)
                n.append(path)

            else:
                for idx, val in enumerate(all_clips):
                    waveform = val.unsqueeze(0)
                    # padding, so all lengths are 319925
                    waveform_padded = torch.nn.functional.pad(waveform, (0, 319925 - waveform.size(1)))
                    waveform_padded = waveform_padded.squeeze(0)
                    X.append(waveform_padded)
                    y.append(label)
                    n.append(all_names[idx])
                

        Xs.append(X)
        ys.append(y)
        ns.append(n)
    
    X_train = Xs[0]
    y_train = ys[0]
    n_train = ns[0]

    X_val = Xs[1]
    y_val = ys[1]
    n_val = ns[1]

    X_test = Xs[2]
    y_test = ys[2]
    n_test = ns[2]
    
    print(f'X: {len(X_train), len(X_val), len(X_test)}')
    print(f'y: {len(y_train), len(y_val), len(y_test)}')
    print(f'id: {len(n_train), len(n_val), len(n_test)}')
    
    train_df = pd.DataFrame({'tensor': X_train, 'label': y_train, 'name': n_train})
    val_df = pd.DataFrame({'tensor': X_val, 'label': y_val, 'name': n_val})
    test_df = pd.DataFrame({'tensor': X_test, 'label': y_test, 'name': n_test})

    train_df.to_csv('TRAIN.csv', index=False)
    val_df.to_csv('VAL.csv', index=False)
    test_df.to_csv('TEST.csv', index=False)
    
    return X_train, y_train, X_val, y_val, X_test, y_test

# With repeating instead of zero padding
def CustomDatasetSplitRepeat(csv_path):
    
    # train val test splitting
    dfs = train_val_test(csv_path)

    base_length = 319925

    # split waveforms if needed, and pad
    Xs = []
    ys = []
    ns = []
    for df in dfs:
        X = []
        y = []
        n = []
        for idx, row in df.iterrows():
            path = row['path']
            label = row['label']
    
            all_clips, all_names = split_audios_sec(path)
            if len(all_clips)==0:
                waveform, _ = torchaudio.load(path)
                waveform = waveform.squeeze(0)
                # padding with repeat, so all lengths are 319925
                
                num_repetitions = base_length // len(waveform)
                waveform_padded = torch.cat([waveform] * num_repetitions, dim=0)
                remaining_length = base_length % len(waveform)
                if remaining_length > 0:
                    waveform_padded = torch.cat([waveform_padded, waveform[:remaining_length]], dim=0)
                
                X.append(waveform_padded)
                y.append(label)
                n.append(path)

            else:
                for idx, val in enumerate(all_clips):
                    # padding, so all lengths are 319925
                    num_repetitions = base_length // len(waveform)
                    waveform_padded = torch.cat([waveform] * num_repetitions, dim=0)
                    remaining_length = base_length % len(waveform)
                    if remaining_length > 0:
                        waveform_padded = torch.cat([waveform_padded, waveform[:remaining_length]], dim=0)
                    
                    waveform_padded = waveform_padded.squeeze(0)
                    X.append(waveform_padded)
                    y.append(label)
                    n.append(all_names[idx])
                

        Xs.append(X)
        ys.append(y)
        ns.append(n)
    
    X_train = Xs[0]
    y_train = ys[0]
    n_train = ns[0]

    X_val = Xs[1]
    y_val = ys[1]
    n_val = ns[1]

    X_test = Xs[2]
    y_test = ys[2]
    n_test = ns[2]
    
    print(f'X: {len(X_train), len(X_val), len(X_test)}')
    print(f'y: {len(y_train), len(y_val), len(y_test)}')
    print(f'id: {len(n_train), len(n_val), len(n_test)}')

    '''
    train_df = pd.DataFrame({'tensor': X_train, 'label': y_train, 'name': n_train})
    val_df = pd.DataFrame({'tensor': X_val, 'label': y_val, 'name': n_val})
    test_df = pd.DataFrame({'tensor': X_test, 'label': y_test, 'name': n_test})
    # saving tensors in csvs lead to them becoming strings, so we do not do this at the moment
    train_df.to_csv('TRAIN.csv', index=False)
    val_df.to_csv('VAL.csv', index=False)
    test_df.to_csv('TEST.csv', index=False)
    '''
    
    return X_train, y_train, n_train, X_val, y_val, n_val, X_test, y_test, n_test
    
class CustomDataset(Dataset):
    def __init__(self, root, df):
        self.root = root
        #self.df = pd.read_csv(os.path.join(root, df))
        self.df = pd.read_csv(df)
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        waveform = self.df.loc[idx, 'tensor']        
        label = self.df.loc[idx, 'label']
        
        waveform, _ = torchaudio.load(audio_path)
        waveform_padded = torch.nn.functional.pad(waveform, (0, 319925 - waveform.size(1)))
        waveform = waveform_padded.tolist()
        label = torch.tensor(label)
        waveform = torch.tensor(waveform)
        return waveform, label


class CustomDataset2(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        waveform = torch.Tensor(self.X[idx])
        label = torch.Tensor([self.y[idx]])
        return waveform, label    