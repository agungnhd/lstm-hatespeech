import pandas as pd

read_file = pd.read_excel (r'raw-data/new_dataset.xlsx')
read_file.to_csv (r'raw-data/new_dataset.csv', index = None, header=["text", "sentiment"])