import os
import yfinance as yf

finance_symbols = [
    'HSBC', 'JPM', 'BAC', 'WFC', 'LYG', 'SPGI', 'MS', 'AXP', 'TD', 'GS', 'HDB', 'BCS', 'MUFG',
    'SCHW', 'MMC', 'BLK', 'PGR', 'CB', 'UBS', 'C', 'IBN', 'CME', 'BX', 'AON', 'SMFG',
    'ICE', 'MCO', 'SAN', 'ITUB', 'KKR', 'AJG', 'USB', 'BNS', 'PFH', 'BBVA', 'ING', 'APO', 
    'PNC', 'AFL', 'AIG', 'MET', 'BSBR', 'MFG', 'COF', 'TFC', 'TRV', 'BK', 'MFC', 'ALL', 'AMP',
    'BBD', 'PRU', 'ARES', 'CSGP', 'ACGL', 'PUK', 'NDAQ', 'NU', 'BBDO', 'SLF', 'HBANM', 'WTW', 
    'HBANP', 'DB', 'EFX', 'COIN', 'CBRE', 'HIG', 'IX', 'NWG', 'TW', 'TROW', 'DFS', 'RJF', 'STT',
    'BRO', 'FCNCA', 'CBOE', 'MTB', 'INVH', 'OWL', 'BEKE', 'ROL', 'MKL', 'WRB', 'LPLA', 'FITBI',
    'FITB', 'RYAN', 'EG', 'PFG', 'FITBP', 'ASBA', 'RKT', 'KB', 'CINF', 'SLMBP', 'HBAN', 'L'
]

if not os.path.exists('./data'):
    os.makedirs('./data')
    os.makedirs('./data/raw')

for stock_id in finance_symbols:
    info = yf.Ticker(stock_id)
    df = info.history(period = 'max')
    df.to_csv(f'./data/raw/{stock_id}_raw.csv')

