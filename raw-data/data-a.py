import numpy as np
import pandas as pd

data = pd.read_csv('raw-data/data_hs/re_dataset.csv', encoding='latin-1')
data = data.head(20)


print("Shape: ", data.shape)
print(data.HS.value_counts())