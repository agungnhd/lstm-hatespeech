import pandas as pd

read_file = pd.read_excel (r'data/clean_trainx.xlsx')
read_file.to_csv (r'data/clean_trainx-csv.csv', index = None, header=None)