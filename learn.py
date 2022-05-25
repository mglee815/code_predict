import numpy as np
import pandas as pd
import torch
import argparse

from torch.utils.data import Dataset, DataLoader
from pytorch_transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from torch.optim import Adam
import torch.nn.functional as F
import pdb
import torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument('--processed_data_path' ,  type = str)
parser.add_argument('--batch_size' ,  type = int)
parser.add_argument('--epochs' ,  type = int)

args = parser.parse_args()


train_df = pd.read_csv(args.processed_data_path + 'train.csv')
test_df = pd.read_csv(args.processed_data_path + 'valid.csv')

class CodeDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        pdb.set_trace()
        text = self.df.iloc[idx, 1]
        label = self.df.iloc[idx, 2]
        return text, label
    
    
code_train_dataset = CodeDataset(train_df)
train_loader = DataLoader(code_train_dataset, batch_size=args.batch_size, shuffle=True)


device = torch.device("cuda")
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased')
model.to(device)

optimizer = Adam(model.parameters(), lr=1e-6)

itr = 1
p_itr = 500
epochs = 1
total_loss = 0
total_len = 0
total_correct = 0


model.train()
for epoch in range(args.epochs):
    

    for idx, (text, label) in enumerate(train_loader):
        if idx == 5 : break
        optimizer.zero_grad()
        
        # encoding and zero padding
        encoded_list = [tokenizer.encode(t, add_special_tokens=True) for t in text]
        padded_list =  [e + [0] * (512-len(e)) for e in encoded_list]
        
        sample = torch.tensor(padded_list)
        sample, label = sample.to(device), label.to(device)
        labels = torch.tensor(label)
        outputs = model(sample, labels=labels)
        loss, logits = outputs

        pred = torch.argmax(F.softmax(logits), dim=1)
        correct = pred.eq(labels)
        total_correct += correct.sum().item()
        total_len += len(labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        
        if itr % p_itr == 0:
            print('[Epoch {}/{}] Iteration {} -> Train Loss: {:.4f}, Accuracy: {:.3f}'.format(epoch+1, epochs, itr, total_loss/p_itr, total_correct/total_len))
            total_loss = 0
            total_len = 0
            total_correct = 0

        itr+=1
        
        
        