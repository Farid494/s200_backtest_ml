import pandas as pd

from datetime import datetime




asset_list = ['GBPUSD']




for asset in asset_list:
    df = pd.read_csv('/Users/apple/Desktop/Forex Amir bhai/s200_backtesting_ML/GU-M1 01.11.2023 - 31.10.2024.csv')
    print('Reading ', asset)
    df.rename(columns={'<TICKER>': 'Ticker', '<DTYYYYMMDD>': 'Date','<TIME>': 'time', '<OPEN>': 'open','<HIGH>': 'high', '<LOW>': 'low', '<CLOSE>' : 'close', '<VOL>' : 'vol'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')

    df['time'] = df['time'].astype(str).str.zfill(4)
    df['time'] = df['time'].str[:2] + ':' + df['time'].str[2:] 
    df['time'] =  pd.to_datetime(df['time'],format='%H:%M')
    df['time'] = pd.to_datetime(df['time'],format).apply(lambda x: x.time())

    # Convert 'Date' to a string in 'YYYY-MM-DD' format
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')

    # Convert 'time' to string format (e.g., 'HH:MM:SS')
    df['time'] = df['time'].astype(str)

    # Now combine 'Date' and 'time' and convert to datetime
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['time'])

    file_path = asset + '.csv'
    df.to_csv(file_path, index=False)
    print("CSV Saved on CSVs Folder")
    print()





