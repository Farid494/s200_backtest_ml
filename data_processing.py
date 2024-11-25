import pandas as pd
from datetime import datetime





asset_list = ['GBPUSD']



import pandas as pd
from datetime import datetime





asset_list = ['GBPUSD']



import pandas as pd

# Replace with the actual path to your CSVs
csv_path = '/Users/apple/Desktop/Forex Amir bhai/s200_backtesting_ML'
output_path = '/Users/apple/Desktop/Forex Amir bhai/s200_backtesting_ML'

for asset in asset_list:
    # Load the CSV file
    df = pd.read_csv(f'{csv_path}/{asset}_1T.csv')
   

    # Ensure 'datetime' is the correct type and set it as the index

    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)

    # The 'open' price will be the 'close' of the previous period
    df['open'] = df['close'].shift(1)
    
    # The first 'open' value will be NaN because there's no previous 'close', so fill it with the first 'close'
    df['open'].fillna(df['close'].iloc[0], inplace=True)

    # Define the aggregation dictionary
    aggregation_dict = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "vol": "sum"
    }

    # Resample the data to 5-minute frequency
    resampled_df = df.resample('D').agg(aggregation_dict)
    resampled_df = resampled_df[resampled_df['vol'] != 0]

    # Save the filtered, resampled data to a new CSV
    resampled_df.to_csv(f'{output_path}/{asset}_D.csv')

 

        

        
