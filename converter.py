import os
import pandas as pd

if not os.path.exists('./data/img'):
    os.makedirs('./data/img')

for stock in os.listdir('./data/raw'):
    df = pd.read_csv(f'./data/raw/{stock}')
    img = df[['Open', 'Close']].apply(lambda x: 1 if x['Close'] > x['Open'] else 0, axis = 1)
    stock_name = stock.split('_')[0]
    img.to_csv(f'./data/img/{stock_name}_img.csv', index = False)