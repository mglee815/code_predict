import numpy as np
import pandas as pd
import torch
import argparse
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import  AutoTokenizer,  AutoConfig, AutoModelForSequenceClassification
from torch.optim import Adam
import torch.nn.functional as F
import pdb
import torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument('--processed_data_path' ,  type = str)
parser.add_argument('--batch_size' ,  type = int)
parser.add_argument('--test_batch_size' ,  type = int)
parser.add_argument('--epochs' ,  type = int)
parser.add_argument('--model_name' ,  type = str)


args = parser.parse_args()


train_df = pd.read_csv(args.processed_data_path + 'train.csv')
dev_df = pd.read_csv(args.processed_data_path + 'valid.csv')
test_df = pd.read_csv(args.processed_data_path + 'valid.csv')


class CodeDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.iloc[idx, 1]
        label = self.df.iloc[idx, 2]
        return text, label
    



code_train_dataset = CodeDataset(train_df)
code_dev_dataset = CodeDataset(dev_df)
code_test_dataset = CodeDataset(test_df)


train_loader = DataLoader(code_train_dataset, batch_size=args.batch_size, shuffle=True)
dev_loader = DataLoader(code_dev_dataset, batch_size=args.test_batch_size, shuffle=True)
test_loader = DataLoader(code_test_dataset, batch_size=args.test_batch_size, shuffle=True)



print("load the model")
print('model name : ' + args.model_name)
device = torch.device("cuda")
config = AutoConfig.from_pretrained(args.model_name)
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
model = AutoModelForSequenceClassification.from_config(config)

### use multi gpu
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model) # multi-gpu training
else:
    model = model.to(device)
        
        
model.to(device)
print("init model done")

optimizer = Adam(model.parameters(), lr=1e-6)




if __name__ == '__main__':
    itr = 1
    p_itr = 10
    epochs = 1
    total_loss = 0
    total_len = 0
    total_correct = 0
    
    
    for epoch in range(args.epochs):
        
        model.train()
        total_loss = 0
        total_len = 0
        total_correct = 0
        for idx, (text, label) in tqdm(enumerate(train_loader)):
            if idx ==10 :break
            
            optimizer.zero_grad()
            # encoding and zero padding
            encoded_list = [tokenizer.encode(t, add_special_tokens=True, max_length = 512) for t in text]
            padded_list =  [e + [0] * (512-len(e)) for e in encoded_list]
            
            sample = torch.tensor(padded_list)
            sample, label = sample.to(device), label.to(device)
            labels = torch.tensor(label)
            outputs = model(sample, labels=labels)
            loss, logits = outputs['loss'].mean(), outputs['logits']
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
        model.eval()

        total_count =0 
        total_correct = 0
        for idx, (text, label) in  tqdm(enumerate(dev_loader), total = len(dev_loader)):
            if idx ==5 :break
            encoded_list = [tokenizer.encode(t, add_special_tokens=True,  max_length = 512) for t in text]
            padded_list =  [e + [0] * (512-len(e)) for e in encoded_list]
            sample = torch.tensor(padded_list)
            sample, label = sample.to(device), label.to(device)
            labels = torch.tensor(label)
            outputs = model(sample, labels=labels)
            loss, logits = outputs['loss'], outputs['logits']
            pred = torch.argmax(F.softmax(logits), dim=1)
            correct = pred.eq(labels)
            total_len += len(labels)
            total_correct += correct.sum().item()
        
        print(f"devaccuracy : {total_correct*100/total_len}%")
    print("Make submission file")
    
    preds = []
    for idx, (text, label) in  tqdm(enumerate(test_loader), total = len(test_loader)):
        encoded_list = [tokenizer.encode(t, add_special_tokens=True, max_length = 512)[:512] for t in text]
        padded_list =  [e + [0] * (512-len(e)) for e in encoded_list]
        sample = torch.tensor(padded_list)
        sample, label = sample.to(device), label.to(device)
        labels = torch.tensor(label)
        outputs = model(sample, labels=labels)
        loss, logits = outputs['loss'], outputs['logits']
        preds += torch.argmax(F.softmax(logits), dim=1).cpu().tolist()
    
    try:
        submission_df = pd.read_csv(args.processed_data_path + 'sample_submission.csv')
        submission_df['similar'] = preds
        submission_df.to_csv(args.processed_data_path + 'submission.csv')
    except:
        pdb.set_trace()
        

    

        

            
            
            
            