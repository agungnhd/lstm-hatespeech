import pandas as pd



data = pd.read_csv('raw-data/data_hs/re_dataset.csv', encoding='latin-1')

columns = ['Abusive','HS_Individual','HS_Group','HS_Religion','HS_Race','HS_Physical','HS_Gender','HS_Other','HS_Weak','HS_Moderate','HS_Strong']
data = data.drop(columns, axis='columns')
data = data.rename(columns={'Tweet': 'text', 'HS': 'sentiment'})

data.to_excel('raw-data/new_dataset.xlsx')