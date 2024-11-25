#SL at third candle 
import json
import datetime
import pandas as pd
import numpy as np
from datetime import time





pairs = ['GBPUSD']

risk_reward_ratios = [1]



# Define additional metrics
def calculate_metrics(trade_results_df):
    gross_profit = trade_results_df[trade_results_df['profit'] > 0]['profit'].sum()
    gross_loss = trade_results_df[trade_results_df['profit'] <= 0]['profit'].sum()
    net_profit = gross_profit + gross_loss

    total_trades = len(trade_results_df)
    long_trades = trade_results_df[trade_results_df['order_type'] == 'buy']
    short_trades = trade_results_df[trade_results_df['order_type'] == 'sell']

    long_net_result = long_trades['profit'].sum()
    short_net_result = short_trades['profit'].sum()

    long_trades_count = len(long_trades)
    short_trades_count = len(short_trades)

    long_win_percentage = len(long_trades[long_trades['profit'] > 0]) / long_trades_count * 100 if long_trades_count > 0 else 0
    short_win_percentage = len(short_trades[short_trades['profit'] > 0]) / short_trades_count * 100 if short_trades_count > 0 else 0

    # max_consecutive_wins = max_consecutive_counts(trade_results_df['profit'] > 0)
    # max_consecutive_losses = max_consecutive_counts(trade_results_df['profit'] <= 0)

    max_consecutive_wins , max_consecutive_losses = max_consecutive_wins_losses(trade_results_df['profit'])

    # Calculate Relative Drawdown
    balance_series = trade_results_df['profit'].cumsum() + initial_balance
    running_max = balance_series.cummax()
    drawdown = running_max - balance_series
    relative_drawdown = round(drawdown.max(),2)

    max_relative_drawdown = round(balance_series.min() - initial_balance)


    # Day of the Week Analysis
    day_of_week_metrics_long = group_and_calculate(trade_results_df, 'TDOW', 'buy')
    day_of_week_metrics_short = group_and_calculate(trade_results_df, 'TDOW', 'sell')

    hour_of_day_metrics_long = group_and_calculate(trade_results_df, 'THOD', 'buy')
    hour_of_day_metrics_short = group_and_calculate(trade_results_df, 'THOD', 'sell')

    date_of_month_metrics_long = group_and_calculate(trade_results_df, 'TDOM', 'buy')
    date_of_month_metrics_short = group_and_calculate(trade_results_df, 'TDOM', 'sell')

    metrics = {
        'Gross Profit': round(gross_profit,2),
        'Gross Loss':  round(gross_loss,2),
        'Net Profit':  round(net_profit,2),
        'Total Trades': total_trades,
        'Long Trades': long_trades_count,
        'Short Trades': short_trades_count,
        'Long Position Net Result': f"${round(long_net_result)} / {round(long_win_percentage,2)}%",
        'Short Position Net Result': f"${round(short_net_result)} / {round(short_win_percentage,2)}%",
        'Maximum Consecutive Wins': f"{max_consecutive_wins}",
        'Maximum Consecutive Loses': f"{max_consecutive_losses}",
        'Relative Drawdown': f"${relative_drawdown}",
        'Max Relative DrawDown': f"${max_relative_drawdown}",    
        'Long Position Metrics by Day of Week':day_of_week_metrics_long,
        'Short Position Metrics by Day of Week:':day_of_week_metrics_short,
        'Long Position Metrics by Hour of Day:':hour_of_day_metrics_long,
        'Short Position Metrics by Hour of Day:':hour_of_day_metrics_short,
        'Long Position Metrics by Date of Month:':date_of_month_metrics_long,
        'Short Position Metrics by Date of Month:':date_of_month_metrics_short,
    }


    return metrics

def check_signal(row, df):
    if row.name >= 3:
        if row['CLOSE'] > df['HIGH'][row.name-3:row.name].max():
            return "BUY"
        elif row['CLOSE'] < df['LOW'][row.name-3:row.name].min():
            return "SELL"
    return ''
               
# Backtesting function

def backtest_trades(historical_data, strategy_results, initial_balance, risk_per_trade, stoploss_type ):
    print('Evaluating Pair:', pair, 'and R:R', risk_reward_ratio, 'with SL', stoploss_type)

    trade_results = []
    end_of_trading_time = datetime.time(23, 58)
    end_of_trading_time_on_friday = datetime.time(20, 58)
    balance = initial_balance

    for index, trade in strategy_results.iterrows():
        trade_date = trade['DATETIME'].date()
        trade_time = trade['DATETIME'].time()
        daily_data = historical_data[historical_data['DATETIME'].dt.date == trade_date]

        # Find the index in historical_data corresponding to the strategy trade time
        trade_index = daily_data[daily_data['DATETIME'].dt.time == trade_time].index

        if not trade_index.empty:
            trade_index = trade_index[0]  # Assuming there's only one match, take the first
            volume = calculate_volume(trade['Entry_Price'], trade['Stop_Loss'], balance, risk_per_trade)
            # Start iterating from the trade_index until a condition is met to close the trade
            for _, row in daily_data.loc[trade_index:].iterrows():
                trade_executed = False
                profit = 0
                close_datetime = None
                exit_price = None
            

                hit_tp = row['HIGH'] >= trade['Take_Profit'] if trade['Signal'] == 'BUY' else row['LOW'] <= trade['Take_Profit']
                hit_sl = row['LOW'] <= trade['Stop_Loss'] if trade['Signal'] == 'BUY' else row['HIGH'] >= trade['Stop_Loss']

                if hit_tp or hit_sl:
                    exit_price = trade['Take_Profit'] if hit_tp else trade['Stop_Loss']
                    profit = (exit_price - trade['Entry_Price']) * volume if trade['Signal'] == 'BUY' else (trade['Entry_Price'] - exit_price) * volume
                    close_datetime = row['DATETIME']
                    trade_executed = True
                    break

                # If the trade was not executed and time passed beyond the last candle of the day, force closure at last price and separate condition for friday
                if not trade_executed and row['DATETIME'].time() == end_of_trading_time or (not trade_executed and row['DATETIME'].time() == end_of_trading_time_on_friday  and trade['DATETIME'].weekday() + 1 == 5):  # Checking if this is the last row in the daily_data ir it is friday
                    exit_price = row['CLOSE']
                    profit = (exit_price - trade['Entry_Price']) * volume if trade['Signal'] == 'BUY' else (trade['Entry_Price'] - exit_price) * volume
                    close_datetime = row['DATETIME']
                    trade_executed = True
                    break
          
            print(trade['DATETIME'])
   
            trade_results.append({
                'open_datetime': trade['DATETIME'],
                'open_price': trade['Entry_Price'],
                'order_type': 'buy' if trade['Signal'] == 'BUY' else 'sell',
                'volume': volume,
                'sl': trade['Stop_Loss'],
                'tp': trade['Take_Profit'],
                'close_datetime': close_datetime,
                'close_price': exit_price,
                'profit': profit,
                'status': 'closed',
                'day': trade['DATETIME'].strftime('%A'),
                'TDOW': trade['DATETIME'].weekday() + 1,
                'THOD': trade['DATETIME'].hour,
                'TDOM': trade['DATETIME'].day,
            })
            balance += profit

    results_df = pd.DataFrame(trade_results)
    total_profit = results_df['profit'].sum()
    win_count = len(results_df[results_df['profit'] > 0])
    loss_count = len(results_df[results_df['profit'] <= 0])
    win_rate = ((win_count)/(win_count + loss_count)) if (win_count + loss_count) > 0 else 0

    return results_df, total_profit, win_count, loss_count, balance, win_rate


def load_seasonality_data(file_path, sheet_name):
    return pd.read_excel(file_path, sheet_name=sheet_name)

def is_cross_pair(pair):
    return 'USD' not in pair 

# Define the True Range function
def true_range(HIGH, LOW, close):
    # The true range is the largest of the:
    # Current HIGH minus the current LOW,
    # Absolute value of the current HIGH minus the previous close,
    # Absolute value of the current LOW minus the previous close.
    tr = HIGH - LOW
    tr = np.maximum(tr, abs(HIGH - close.shift()))
    tr = np.maximum(tr, abs(LOW - close.shift()))
    return tr 


def place_trades(df, sub_df, daily_atr, previous_9pm_open):
    trades = []
    buy_price_point = previous_9pm_open + 0.25 * daily_atr  
    sell_price_point = previous_9pm_open - 0.25 * daily_atr

    for i in range(2, len(sub_df)):

        signal = None
        condition_a = None
        condition_b = None

        # Calculate the movement of the last three M5 candles
        high_max = sub_df['HIGH'].iloc[i-2:i+1].max()
        low_min = sub_df['LOW'].iloc[i-2:i+1].min()
        candle_move = high_max - low_min

        

        # Entry Pattern Check: Candle move within specified ATR range
        if 0.10 * daily_atr < candle_move < 0.20 * daily_atr:
            all_positive = ((sub_df['CLOSE'].iloc[i-2] > sub_df['OPEN'].iloc[i-2]) and \
               (sub_df['CLOSE'].iloc[i-1] > sub_df['OPEN'].iloc[i-1]) and 
               (sub_df['CLOSE'].iloc[i] > sub_df['OPEN'].iloc[i]))
            
            all_negative = ((sub_df['CLOSE'].iloc[i-2] < sub_df['OPEN'].iloc[i-2]) and \
               (sub_df['CLOSE'].iloc[i-1] < sub_df['OPEN'].iloc[i-1]) and \
               (sub_df['CLOSE'].iloc[i] < sub_df['OPEN'].iloc[i]))

            
            # Condition A: Making higher highs in a bullish scenario and lower lows in a bearish scenario
            if all_positive:
                signal = 'BUY'
                #E1
                # condition_a = (sub_df['CLOSE'].iloc[i] > sub_df['OPEN'].iloc[i] and sub_df['CLOSE'].iloc[i-1] > sub_df['OPEN'].iloc[i-1] and sub_df['CLOSE'].iloc[i-2] > sub_df['OPEN'].iloc[i-2])

                #E2
                 # All candles are bullish and each closes higher than the previous one
                # condition_a = (sub_df['CLOSE'].iloc[i] > sub_df['OPEN'].iloc[i] and
                #             sub_df['CLOSE'].iloc[i-1] > sub_df['OPEN'].iloc[i-1] and
                #             sub_df['CLOSE'].iloc[i-2] > sub_df['OPEN'].iloc[i-2] and
                #             sub_df['CLOSE'].iloc[i] > sub_df['CLOSE'].iloc[i-1] and
                #             sub_df['CLOSE'].iloc[i-1] > sub_df['CLOSE'].iloc[i-2])
                
                #E3
                # condition_a = sub_df['HIGH'].iloc[i] > sub_df['HIGH'].iloc[i-1] and sub_df['HIGH'].iloc[i-1] > sub_df['HIGH'].iloc[i-2]

                #E4
                # Calculate the absolute body size of each candle
                body_candle_1 = abs(sub_df['CLOSE'].iloc[i-2] - sub_df['OPEN'].iloc[i-2])
                body_candle_2 = abs(sub_df['CLOSE'].iloc[i-1] - sub_df['OPEN'].iloc[i-1])
                body_candle_3 = abs(sub_df['CLOSE'].iloc[i] - sub_df['OPEN'].iloc[i])

                # Calculate the average body size of Candle #1 and #2
                average_body_12 = (body_candle_1 + body_candle_2) / 2
                condition_a = body_candle_3 >= 0.8 * average_body_12 and (sub_df['CLOSE'].iloc[i] > sub_df['OPEN'].iloc[i] and sub_df['CLOSE'].iloc[i-1] > sub_df['OPEN'].iloc[i-1] and sub_df['CLOSE'].iloc[i-2] > sub_df['OPEN'].iloc[i-2])

                condition_b = (sub_df['HIGH'].iloc[i] < buy_price_point)

            elif all_negative:
                signal = 'SELL'
                #E1
                # condition_a = (sub_df['CLOSE'].iloc[i] < sub_df['OPEN'].iloc[i] and sub_df['CLOSE'].iloc[i-1] < sub_df['OPEN'].iloc[i-1] and sub_df['CLOSE'].iloc[i-2] < sub_df['OPEN'].iloc[i-2])

                #E2
                # condition_a = (sub_df['CLOSE'].iloc[i] < sub_df['OPEN'].iloc[i] and
                # sub_df['CLOSE'].iloc[i-1] < sub_df['OPEN'].iloc[i-1] and
                # sub_df['CLOSE'].iloc[i-2] < sub_df['OPEN'].iloc[i-2] and
                # sub_df['CLOSE'].iloc[i] < sub_df['CLOSE'].iloc[i-1] and
                # sub_df['CLOSE'].iloc[i-1] < sub_df['CLOSE'].iloc[i-2])


                #E3
                # condition_a = sub_df['LOW'].iloc[i] < sub_df['LOW'].iloc[i-1] and sub_df['LOW'].iloc[i-1] < sub_df['LOW'].iloc[i-2]

                #E4
                # Calculate the absolute body size of each candle
                body_candle_1 = abs(sub_df['CLOSE'].iloc[i-2] - sub_df['OPEN'].iloc[i-2])
                body_candle_2 = abs(sub_df['CLOSE'].iloc[i-1] - sub_df['OPEN'].iloc[i-1])
                body_candle_3 = abs(sub_df['CLOSE'].iloc[i] - sub_df['OPEN'].iloc[i])

                # Calculate the average body size of Candle #1 and #2
                average_body_12 = (body_candle_1 + body_candle_2) / 2
                condition_a = body_candle_3 >= 0.8 * average_body_12 and (sub_df['CLOSE'].iloc[i] < sub_df['OPEN'].iloc[i] and sub_df['CLOSE'].iloc[i-1] < sub_df['OPEN'].iloc[i-1] and sub_df['CLOSE'].iloc[i-2] < sub_df['OPEN'].iloc[i-2])

                condition_b = (sub_df['LOW'].iloc[i] > sell_price_point)
        
            if condition_a and condition_b and signal and i + 1 < len(sub_df):

                
                # Set entry price based on signal and predefined retracement
                
                entry_price = sub_df['OPEN'].iloc[i+1]

                #SL1
                # Stop loss setup
                # stop_loss = (entry_price - 0.30 * daily_atr) if signal == 'BUY' else (entry_price + 0.30 * daily_atr) 

                #SL2
                mid_point_candle_2 = (sub_df['HIGH'].iloc[i-1] + sub_df['LOW'].iloc[i-1]) / 2  #
                stop_loss = mid_point_candle_2 

                #SL3
                # pip_size = 0.00020 if pair in ['GBPUSD', 'USDCAD']  else 0.0002
                # if signal == 'BUY':
                #     stop_loss = sub_df['LOW'].iloc[i] - pip_size 
                # else:
                #     stop_loss = sub_df['HIGH'].iloc[i] + pip_size 


                # Define the trade
                risk_amount = abs(entry_price - stop_loss)
                take_profit = entry_price + (risk_reward_ratio * risk_amount) if signal == 'BUY' else entry_price - (risk_reward_ratio * risk_amount)
                price_point_adjusted = buy_price_point if signal == 'BUY' else sell_price_point
                trades.append({
                    'DATETIME': sub_df['DATETIME'].iloc[i+1],
                    'Signal': signal,
                    'Entry_Price': entry_price,
                    'Stop_Loss': stop_loss,
                    'Take_Profit': take_profit,
                    'daily_atr': daily_atr,
                    'candle_move': candle_move,
                    'previous_9pm_open': previous_9pm_open,
                    'price_point_adjusted':price_point_adjusted,
                    'Condition_A': condition_a,
                    'Condition_B': condition_b,
                })

                # Optional: Mark the signal and ATR conditions in the original dataframe
                df.loc[sub_df.index[i], 'Signal'] = signal
                df.loc[sub_df.index[i], 'ATR_Range'] = f"{0.20 * daily_atr}-{0.40 * daily_atr}"
                
    return trades


def compute_atr(df):
        df['TR'] = (df['HIGH'] - df['LOW']).round(4)
        return df

# Function to calculate the volume based on risk management
def calculate_volume(entry_price, stop_loss, balance, risk_per_trade):
    risk_amount = balance * risk_per_trade
    sl_distance = abs(entry_price - stop_loss)
    if sl_distance > 0:
        volume = risk_amount / sl_distance
    else:
        volume = 0
    return volume

def group_and_calculate(df, group_by_column, position_type=None):
        if position_type:
            df = df[df['order_type'] == position_type]
        
        grouped = df.groupby(group_by_column).agg(
            net_profit=('profit', 'sum'),  
            trade_count=('profit', 'count')
        ).reset_index()

        return grouped
        
def max_consecutive_wins_losses(profits):
    max_wins = 0
    max_losses = 0
    current_wins = 0
    current_losses = 0
    
    for profit in profits:
        if profit > 0:  # It's a win
            current_wins += 1
            current_losses = 0  # Reset losses count
        elif profit < 0:  # It's a loss
            current_losses += 1
            current_wins = 0  # Reset wins count
        else:  # If no profit or loss, reset both
            current_wins = 0
            current_losses = 0
        
        # Check if the current streak is the longest
        if current_wins > max_wins:
            max_wins = current_wins
        if current_losses > max_losses:
            max_losses = current_losses

    return max_wins, max_losses



def fetch_current_day_open(df, current_date):
    """Fetch the current day's opening price."""
    try:
        # Ensure current_date is a Timestamp for consistent DataFrame indexing
        current_date = pd.Timestamp(current_date)
        return df.at[current_date, 'OPEN']
    except KeyError:
        # Return None if there is no data for that date
        return None



for pair in pairs:

    summary_text =''
    
    #Data Trim

    # Daily 1000 for 5 years
    # 5 min 300000
    # Read daily data
    df_daily = pd.read_csv(f"/Users/apple/Desktop/Forex Amir bhai/s200_backtesting_ML/GBPUSD_b_D1.csv")
    df_last_100_daily = df_daily    


    # Change headers to uppercase
    df_last_100_daily.columns = map(str.upper, df_last_100_daily.columns)

    # Print and save to CSV
    # print(df_last_100_daily)
    daily_trimmed_output_file = f'/Users/apple/Desktop/Forex Amir bhai/s200_backtesting_ML/{pair}_b_D_trimmed.csv'
    df_last_100_daily.to_csv(daily_trimmed_output_file, index=False)

    # Read 5-minute data
    df_1min = pd.read_csv(f'/Users/apple/Desktop/Forex Amir bhai/s200_backtesting_ML/GBPUSD_b_M1.csv')
    df_last_100_1min = df_1min

    # Change headers to uppercase
    df_last_100_1min.columns = map(str.upper, df_last_100_1min.columns)

    # Print and save to CSV
    # print(df_last_100_5min)
    min1_trimmed_output_file = f'/Users/apple/Desktop/Forex Amir bhai/s200_backtesting_ML/{pair}_1T_trimmed.csv'
    df_last_100_1min.to_csv(min1_trimmed_output_file, index=False)


    # Part 1: Initial Setup and Data Loading
    # **************************************************
    # Read the CSV data into a DataFrame
    file_path_daily_data = daily_trimmed_output_file
    file_path_1_min_data = min1_trimmed_output_file

    data = pd.read_csv(file_path_daily_data)

    # Calculate True Range for each candle
    data['TR'] = true_range(data['HIGH'], data['LOW'], data['CLOSE'])

    # Calculate the Average True Range (ATR)
    data['ATR'] = round(data['TR'].rolling(window=14, min_periods=1).mean(), 5)


    data['Signal'] = data.apply(check_signal, axis=1, df=data)
    data['Signal'] = data['Signal'].shift(+1)



    daily_signal_path = f'/Users/apple/Desktop/Forex Amir bhai/s200_backtesting_ML/DAILY_{pair}_D_SIGNAL.csv'
    # Save the DataFrame to a new CSV file
    data.to_csv(daily_signal_path, index=False)

        # print(data)
        # Part 2: Generate Trading Signals

        # Read and preprocess daily data
  

    for risk_reward_ratio in risk_reward_ratios:
        daily_df = pd.read_csv(daily_signal_path)
        daily_df.dropna(subset=['DATETIME'], inplace=True)


        # Read and preprocess 1 min data
        five_min_df = pd.read_csv(file_path_1_min_data)
        five_min_df['DATETIME'] = five_min_df['DATETIME'] = pd.to_datetime(five_min_df['DATETIME'], format='%d/%m/%Y %H:%M', errors='coerce')
        five_min_df.dropna(subset=['DATETIME'], inplace=True)


        five_min_groups = five_min_df.groupby(five_min_df['DATETIME'].dt.date) 

        daily_df['DATETIME'] = pd.to_datetime(daily_df['DATETIME'], dayfirst=True)

        daily_df.set_index('DATETIME', inplace=True)

        all_trades = []

     
    

        for index, row in daily_df.iterrows():
            date = index.date()
            cuurent_day_open_price = fetch_current_day_open(daily_df, date)
            previous_row = daily_df.iloc[daily_df.index.get_loc(index) - 1]
            daily_atr = previous_row['ATR']

    
            
            if date in five_min_groups.groups and cuurent_day_open_price:
                sub_df = five_min_groups.get_group(date)
                # sub_df['DATETIME'] = pd.to_datetime(sub_df['DATETIME'])   ###
                sub_df.set_index('DATETIME', inplace=True)
                sub_df = sub_df.between_time('08:00:00', '20:00:00').reset_index()
                
                trades = place_trades(five_min_df, sub_df, daily_atr, cuurent_day_open_price)
                all_trades.extend(trades)
                
  
        if not all_trades:
            continue
        trades_df = pd.DataFrame(all_trades)
        results_file_path = f'/Users/apple/Desktop/Forex Amir bhai/s200_backtesting_ML/{pair}_trade_results.csv'
        trades_df.to_csv(results_file_path, index=False)
     

        # Part 3: Backtest Results Analysis

        # Load the historical 5-minute candle data
        historical_data = pd.read_csv(file_path_1_min_data)
        historical_data['DATETIME'] = pd.to_datetime(historical_data['DATETIME'], format="%d/%m/%Y %H:%M", errors='coerce')
        # historical_data['DATETIME'] = pd.to_datetime(historical_data['DATETIME'], format="%d/%m/%Y %H:%M")

        # Load your strategy results
        strategy_results = pd.read_csv(results_file_path)
        strategy_results['DATETIME'] = pd.to_datetime(
            strategy_results['DATETIME'], format="%Y-%m-%d %H:%M:%S")

        # Define the initial balance and risk per trade
        initial_balance = 100000
        risk_per_trade = 0.0025  # 0.25%

        # stoplosses = ['Fixed SL', 'Breakeven SL', "Trailing SL"]

        stoplosses = ['Fixed SL']
        for stoploss in stoplosses:
        # Perform the backtest
            trade_results_df, total_profit, win_count, loss_count, ending_balance, win_rate = backtest_trades(
                historical_data, strategy_results, initial_balance, risk_per_trade, stoploss)
            
            additional_metrics = calculate_metrics(trade_results_df)

            # Display the overall results
            print('Pair: ', pair)
            print(f"Ending Balance: {round(ending_balance,2)}")
            print(f"Total Profit: {round(total_profit,2)}")
            print(f"Total Trades: {win_count+ loss_count}")
            print(f"Win Count: {win_count }")
            print(f"Loss Count: {loss_count}")
            print(f"Win Rate: {round(win_rate * 100, 2)}%")
            print()


            summary_text += (
                "-------------\n"
                f"Evaluating Pair:, {pair} and R:R  {risk_reward_ratio} with SL ,{stoploss}\n"
                f"Pair: : {pair}\n"
                f"Ending Balance: {round(ending_balance, 2)}\n"
                f"Total Profit: {round(total_profit, 2)}\n"
                f"Total Trades: {win_count + loss_count}\n"
                f"Win Count: {win_count}\n"
                f"Loss Count: {loss_count}\n"
                f"Win Rate: {round(win_rate * 100, 2)}%\n"
            )
            for metric, value in additional_metrics.items():
                if isinstance(value, pd.DataFrame):
                    df_string = value.to_string(index=False)
                    summary_text += f"{metric}:\n{df_string}\n\n"

                    print(f"{metric}:")
                    print(value)
                    print()
                else:
                    print(f"{metric}: {value}")
                    summary_text += f"{metric}: {value}\n"

            # Define the file path
            results_file_path = f'/Users/apple/Desktop/Forex Amir bhai/s200_backtesting_ML{pair}_results.txt'

            # print(summary_text)
            # Write the summary text to the file
            with open(results_file_path, 'w') as file:
                file.write(summary_text)

            # Save the detailed trade results to a CSV file
            import pdb
            pdb.set_trace()
            trade_results_df.to_csv(
                f'/Users/apple/Desktop/Forex Amir bhai/s200_backtesting_ML_backtest_results_original_broker_data_{risk_reward_ratio}_{pair}_{stoploss}.csv', index=False)


