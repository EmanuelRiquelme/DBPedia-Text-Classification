import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import BertTokenizer

class dataset(Dataset):
    def __init__(self,csv_file_name,token_len = 128,
                 tokenizer_nombre = 'bert-base-cased'):
        self.df = self.build_df(csv_file_name)
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_nombre)
        self.token_len = token_len
    
    def build_df(self,csv_file_name):
        df = pd.read_csv(csv_file_name)
        df['l1'] = df['l1'].astype('category').cat.codes
        df['l2'] = df['l2'].astype('category').cat.codes
        df['l3'] = df['l3'].astype('category').cat.codes
        return df
    def __len__(self):
        return len(self.df)
    def __getitem__(self,idx):
        label = torch.tensor((self.df['l1'].loc[idx],self.df['l2'].loc[idx],self.df['l3'].loc[idx]))
        txt = self.df['text'][0]
        txt = self.tokenizer(txt , return_tensors="pt",padding='max_length',
                             truncation=True, max_length=self.token_len)
        return txt,label.type(torch.LongTensor)
