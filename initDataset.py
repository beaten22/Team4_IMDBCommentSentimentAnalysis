import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import torch
from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import InputExample

dataset_directory="Dataset/IMDB_Dataset.csv"
df_dataset=pd.read_csv(dataset_directory)
missing_values = df_dataset[df_dataset.isnull().any(axis=1)] # there wasn't missing column
duplicates = df_dataset[df_dataset.duplicated(subset=["review", "sentiment"], keep=False)]

#I dropped duplicated rows from dataset
cleaned_df = df_dataset.drop_duplicates(subset=["review", "sentiment"])
df = cleaned_df.dropna()

print(df.sentiment.value_counts(normalize = True))

# convert categorical target to numeric
def convert_target(value):
    if value=="positive":
        return 1
    else:
        return 0
    
df['sentiment']  =  df['sentiment'].apply(convert_target)

# load model and tokenizer
tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-cased')

# test model
example='Who was Jim Henson ?'
tokens=tokenizer.encode(example, add_special_tokens=True)

print(tokens)
print(tokenizer.decode(tokens))

tokens_tensor = torch.tensor([tokens])

print(tokens_tensor)

