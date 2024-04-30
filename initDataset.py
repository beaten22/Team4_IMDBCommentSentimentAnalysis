import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
dataset_directory="Dataset/IMDB_Dataset.csv"
df_dataset=pd.read_csv(dataset_directory)
missing_values = df_dataset[df_dataset.isnull().any(axis=1)] # there wasn't missing column
duplicates = df_dataset[df_dataset.duplicated(subset=["review", "sentiment"], keep=False)]

#I dropped duplicated rows from dataset
cleaned_df = df_dataset.drop_duplicates(subset=["review", "sentiment"])
cleaned_df = cleaned_df.dropna()