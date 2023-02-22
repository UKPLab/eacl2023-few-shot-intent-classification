from email.policy import default
from enum import unique
import itertools
import os
from collections import defaultdict

import pandas as pd
import torch
from torch.utils.data import Dataset, TensorDataset
import csv

def read_episode_from_file(fileop):
    is_divider = lambda line: line.strip() == ''
    for divider, episode in itertools.groupby(fileop, is_divider):
        if not divider:
            cols = [line.strip().split('|') for line in episode]
            x = [row[0].strip() for row in cols]
            y = [row[1].strip() for row in cols]
            yield (x, y)

def read_episodes_from_directory(data_dir):
    episodes = []
    for fname in sorted(os.listdir(data_dir)):
        with open(os.path.join(data_dir, fname), "r", encoding='utf-8') as f:
            subsets = [subset for subset in read_episode_from_file(f)]
            assert len(subsets) == 2, f"{data_dir/fname} contains {len(episode)} sample sets. Each episode file should contain only one support & query sets!"
            intent_labels = list(set(subsets[0][1]))
            try:
                episode = {
                    'xs': subsets[0][0],
                    'xq': subsets[1][0],
                    'ys_label': subsets[0][1],
                    'yq_label': subsets[1][1],
                    'labels': intent_labels,
                    'ys': [intent_labels.index(y) for y in subsets[0][1]],
                    'yq': [intent_labels.index(y) for y in subsets[1][1]]
                }
            except Exception as e:
                print(f"{e}")
                print(f"file: {data_dir+fname}, solution: remove | from the beginning of any line")
                exit(0)
            episodes.append(episode)
    return episodes
    

class EpisodeDataset(Dataset):
    def __init__(self, data_dir):
        super(EpisodeDataset, self).__init__()
        self.data_dir = data_dir

        # Init
        self.episodes = read_episodes_from_directory(data_dir)
        self.size = len(self.episodes)
    
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.episodes[idx]

class NonEpisodeDataset(Dataset):
    def __init__(self, data_path):
        super(NonEpisodeDataset, self).__init__()
        self.data_path = data_path
        self.sents, self.labels = self.read_data()
        self.size = len(self.sents)

    def read_data(self):
        
        df = pd.read_csv(self.data_path, sep='\t', quoting=csv.QUOTE_NONE, encoding='utf-8')
       
        if len(df.columns) == 3:
            df.columns = ["sent", "slot", "intent"]
        elif len(df.columns) == 2:
            df.columns = ["sent", "intent"]
        else:
            raise NotImplemented("we do not support files whose number of columns is not 2 or 3. ")
            
        sents = df['sent'].values.tolist()
        
        if df.dtypes['intent'] == 'int64':    
            labels = df['intent'].values.tolist()
        else:
            labels = df['intent'].values.tolist()
            
            labels_map = {}
            
            for label in labels:
                if label not in labels_map:
                    labels_map[label] = len(labels_map)
            
            labels = [labels_map[label] for label in labels]
        
        assert (len(sents)==len(labels))
        
        return sents, labels
    
    def __len__(self):
        
        return self.size

    def __getitem__(self, idx):
        
        x = self.sents[idx].lower().strip()
        
        y = self.labels[idx]
        
        sample ={"x":x, "y":y}
        
        return sample


def _test_non_episode():
    
    atis_path = 'dataset_episodes/snips/train/train.csv'
    
    snipts_path = 'dataset_episodes/snips/train/train.csv'
    
    top_path = 'dataset_episodes/fb_top/train/train.csv'
    
    banking77_path = 'dataset_episodes/BANKING77/episodes-few-shot/01/5C_1K/seed42/train_full.csv'
    
    clinic150_path = 'dataset_episodes/OOS/episodes-few-shot/04/5C_5K/seed42/train_full.csv'
    
    hwu64_path = 'dataset_episodes/HWU64/episodes-few-shot/05/5C_1K/seed42/train_full.csv'
    
    path = hwu64_path
    
    nsd = NonEpisodeDataset(path)
    
    for _,item in enumerate(nsd):
        print(item)


def _test_episode():
    path = "dataset_episodes/atis_temp/dev/"
    epd = EpisodeDataset(path)
    for _, item in enumerate(epd):
        print(item)
        break

if __name__== "__main__":
    # _test_non_episode()
    _test_episode()    
    