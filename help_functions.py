import pandas as pd
import os
from IPython.display import Audio
import torch
import torchaudio
from sklearn.model_selection import train_test_split
import librosa
import math

# For long audio files (> 20s), we want to split them into smaller segments of 10s
def split_audios_sec(file_path):
    name = file_path[:-4]
    
    all_clips = []
    all_names = []

    fs = 16000
    waveform,_ = torchaudio.load(file_path)
    len_w = len(waveform[0])
    len_s = len_w/fs
    limit_s = 20 # if length is over 20s, start cutting
    limit_w = limit_s * fs

    cutting_limit_s = 10 # cutting into 10s chunks
    cutting_limit_w = cutting_limit_s * fs

    if len_s >= limit_s:
        for idx in range(0, len_w, cutting_limit_w):
            segment = waveform[0][idx:(idx+cutting_limit_w)]
            len_segment_s = len(segment)/fs
            #print(len_cut_audio_s, cutting_limit_s)
            if len_segment_s < cutting_limit_s:
                all_clips[-1] = torch.cat((all_clips[-1], segment))
            else:
                all_clips.append(segment)
        for n in range(len(all_clips)):
            new_name = name + '_clipped_' + str(n+1)
            all_names.append(new_name)
            
    return all_clips, all_names

def train_val_test(csv_path):
    df = pd.read_csv(csv_path)

    # GETTING UNIQUE IDS
    ids = df['id']
    id_names = []
    for n in ids:
        if n not in id_names:
            id_names.append(n)
    id_train, id_temp = train_test_split(id_names, test_size=0.3, random_state=42)
    id_val, id_test = train_test_split(id_temp, test_size=1/3, random_state=42)

    # GETTING X AND y
    X_train = []
    X_val = []
    X_test = []
    y_train = []
    y_val = []
    y_test = []
    for _, row in df.iterrows():
        id = row['id']
        path = row['path']
        label = row['label']
        
        if id in id_train:
            X_train.append(path)
            y_train.append(label)
        elif id in id_val:
            X_val.append(path)
            y_val.append(label)
        elif id in id_test:
            X_test.append(path)
            y_test.append(label)
        else:
            print('ERROR')

    print(f'X: {len(X_train), len(X_val), len(X_test)}')
    print(f'y: {len(y_train), len(y_val), len(y_test)}')
    print(f'id: {len(id_train), len(id_val), len(id_test)}')
    
    train_df = pd.DataFrame({'path': X_train, 'label': y_train})
    val_df = pd.DataFrame({'path': X_val, 'label': y_val})
    test_df = pd.DataFrame({'path': X_test, 'label': y_test})
    
    return train_df, val_df, test_df