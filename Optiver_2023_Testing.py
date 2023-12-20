# %% [markdown]
# # Optivar Trading at the Close | Testing

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-12-20T21:12:56.491976Z","iopub.execute_input":"2023-12-20T21:12:56.492499Z","iopub.status.idle":"2023-12-20T21:13:11.882382Z","shell.execute_reply.started":"2023-12-20T21:12:56.492452Z","shell.execute_reply":"2023-12-20T21:13:11.881170Z"}}
######################
# Libraries and Data #
######################


# Data Manipulation
import pandas as pd
import numpy as np



# Plotting
import matplotlib.pyplot as plt

# Statistical Operations
from scipy import stats

# File Operations
import pickle


# Machine Learning Models
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

#Neural Net Modelling
import keras.backend as K
import tensorflow as tf
from keras.layers import Input, Dense, LSTM ,Conv1D,Dropout,Bidirectional,Multiply
from keras.models import Model
from keras.layers import Multiply
from keras.models import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Concatenate
from tensorflow.keras.layers import Input, Dense, Conv1D, Dropout, BatchNormalization, LSTM, Bidirectional, Flatten, Concatenate, Permute, Multiply
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split
import numpy as np


#Speed Processing and Parallel computations


import gc  
import os  
import time  
import warnings  
from itertools import combinations  
from warnings import simplefilter  
import logging
from numba import njit, prange
#import cudf
import multiprocessing
# Disable warnings to keep the code clean
warnings.filterwarnings("ignore")
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

with open("/kaggle/input/recentmodel/XGB_dynamicmodel (7).pkl", "rb") as f:
       model = pickle.load(f)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-12-20T21:13:11.884538Z","iopub.execute_input":"2023-12-20T21:13:11.885449Z","iopub.status.idle":"2023-12-20T21:13:12.008276Z","shell.execute_reply.started":"2023-12-20T21:13:11.885405Z","shell.execute_reply":"2023-12-20T21:13:12.007261Z"}}
#API-Connection
import optiver2023
env = optiver2023.make_env()
iter_test = env.iter_test()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-12-20T21:13:12.010160Z","iopub.execute_input":"2023-12-20T21:13:12.010972Z","iopub.status.idle":"2023-12-20T21:13:12.029506Z","shell.execute_reply.started":"2023-12-20T21:13:12.010926Z","shell.execute_reply":"2023-12-20T21:13:12.028341Z"}}
# Function to reduce memory usage of a Pandas DataFrame
def reduce_mem_usage(df, verbose=0):
    """
    Iterate through all numeric columns of a dataframe and modify the data type
    to reduce memory usage.
    """
    
    # Calculate the initial memory usage of the DataFrame
    start_mem = df.memory_usage().sum() / 1024**2

    # terate through each column in the DataFrame
    for col in df.columns:
        col_type = df[col].dtype

        # Check if the column's data type is not 'object' (i.e., numeric)
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            # Check if the column's data type is an integer
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                # Check if the column's data type is a float
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float32)

    # Provide memory optimization information if 'verbose' is True
    if verbose:
        logger = logging.getLogger(__name__)
        logger.info(f"Memory usage of dataframe is {start_mem:.2f} MB")
        end_mem = df.memory_usage().sum() / 1024**2
        logger.info(f"Memory usage after optimization is: {end_mem:.2f} MB")
        decrease = 100 * (start_mem - end_mem) / start_mem
        logger.info(f"Decreased by {decrease:.2f}%")

    return df

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-12-20T21:13:12.034139Z","iopub.execute_input":"2023-12-20T21:13:12.034723Z","iopub.status.idle":"2023-12-20T21:13:12.188497Z","shell.execute_reply.started":"2023-12-20T21:13:12.034683Z","shell.execute_reply":"2023-12-20T21:13:12.186689Z"}}

@njit(fastmath=True, parallel=True)
def rolling_mean(arr, window, min_periods):
    n = len(arr)
    means = np.empty(n, dtype=np.float64)
    stds = np.empty(n, dtype=np.float64)

    for i in prange(n):
        if i < window:
            sub_arr = arr[max(0, i - min_periods + 1):i + 1]
        else:
            sub_arr = arr[i - window + 1:i + 1]
        
        if len(sub_arr) >= min_periods:
            means[i] = np.mean(sub_arr)
            stds[i] = np.nan if len(sub_arr) < 2 else np.std(sub_arr)
        else:
            means[i] = np.nan
            stds[i] = np.nan

    return means, stds

@njit(fastmath=True, parallel=True)
def rolling_mean_target(arr, window, min_periods):
    n = len(arr)
    means = np.empty(n, dtype=np.float64)
    stds = np.empty(n, dtype=np.float64)

    for i in prange(n):
        if i < window:
            # Use data up to but not including the current index
            sub_arr = arr[max(0, i - min_periods + 1):i]
        else:
            # Exclude the current point from the window
            sub_arr = arr[i - window:i]
        
        if len(sub_arr) >= min_periods:
            means[i] = np.mean(sub_arr)
            # Calculate standard deviation, handle cases with less than two elements
            stds[i] = np.std(sub_arr) if len(sub_arr) > 1 else np.nan
        else:
            means[i] = np.nan
            stds[i] = np.nan

    return means, stds



def apply_rolling(df, feature, window_size, min_periods=1):
    means, stds = np.empty(len(df)), np.empty(len(df))
    grouped = df.groupby(['stock_id'])[feature]
    if feature != 'target':
        for group_key, group_values in grouped:
            m, s = rolling_mean(group_values.values, window_size, min_periods)
            means[group_values.index] = m
            stds[group_values.index] = s
    
    elif feature == 'target':
        for group_key, group_values in grouped:
            m,s= rolling_mean_target(group_values.values, window_size, min_periods)
            means[group_values.index] = m
            stds[group_values.index] = s
            
    return means, stds




    
def feature_engineering_train(df):
            # Remove Data with no target value
            df.dropna(subset=['target'], inplace=True)


            df.sort_values(by=['stock_id','date_id','seconds_in_bucket'], inplace=True)
            

            cols_to_interpolate = ['imbalance_size', 'reference_price', 'matched_size', 
                                'bid_price', 'bid_size', 'ask_price', 'ask_size', 'wap', 'imbalance_buy_sell_flag']

            for col in cols_to_interpolate:
                if df[col].isna().any():
                    df[col] = df[col].interpolate(method='cubic')

            
            if df['far_price'].isna().any():
                train_far = df[df['far_price'].notna()]
                X_train_far = train_far[['imbalance_buy_sell_flag', 'reference_price']]
                y_train_far = train_far['far_price']
                
                model_far = LinearRegression()
                model_far.fit(X_train_far, y_train_far)
                
                nan_far_index = df[df['far_price'].isna()].index
                
                df.loc[nan_far_index, 'far_price'] = model_far.predict(df.loc[nan_far_index, ['imbalance_buy_sell_flag', 'reference_price']])
                
                with open('modelfar.pkl', 'wb') as f: 
                    pickle.dump(model_far, f)
            
            if df['near_price'].isna().any():
                train_near = df[df['near_price'].notna()]
                X_train_near = train_near[['bid_price', 'ask_price', 'wap', 'reference_price', 'imbalance_buy_sell_flag']]
                y_train_near = train_near['near_price']

                model_near = LinearRegression()
                model_near.fit(X_train_near, y_train_near)
            
                nan_near_index = df[df['near_price'].isna()].index
                
                df.loc[nan_near_index, 'near_price'] = model_near.predict(df.loc[nan_near_index, ['bid_price', 'ask_price', 'wap', 'reference_price', 'imbalance_buy_sell_flag']])
                
                with open('modelnear.pkl', 'wb') as f: 
                    pickle.dump(model_near, f)
            
            z_scores = np.abs(stats.zscore(df.select_dtypes(include=[np.number])))
            filtered_entries = (np.abs(z_scores) <= 2.5).all(axis=1)
            df = df[filtered_entries]
    


            #df = cudf.from_pandas(df)
            ### Imbalance Features 
            # Imbalance Direction
            df['ImbalanceDirection'] = df['imbalance_size'] * df['imbalance_buy_sell_flag']

            # Book Imbalance
            df['BookImbalance'] = (df['bid_size'] - df['ask_size']) / (df['bid_size'] + df['ask_size'])


            # Price Imbalance
            df['PriceImbalance'] = df['reference_price'] - df['wap']

            

            ### Order Book Features
            # Bid-Ask Spread
            df['BidAskSpread'] = df['ask_price'] - df['bid_price']

            # MidPrice and its Movement
            df['MidPrice'] = (df['ask_price'] + df['bid_price']) / 2
            df['MidPrice_Movement'] = df['MidPrice'].diff()
            
            # Price Spread
            df['PriceSpread'] = df['ask_price'] - df['bid_price']

            # Normalized Spread
            df['NormSpread'] = df['PriceSpread'] / df['wap']


            # Size Ratio
            df['SizeRatio'] = df['bid_size'] / df['ask_size']

            ### Order Flow Features
            # Volumed and volume
            df['volume'] = df['matched_size'].pct_change() * 100
            
            # Volume of the previous day
            df['prev_day_volume'] = df.groupby(['stock_id', 'date_id'])['matched_size'].transform('sum').shift(1)

            # Daily volatility (standard deviation of wap for each day)
            df['daily_volatility'] = df.groupby(['stock_id', 'date_id'])['wap'].transform('std')

            # Shift the daily volatility to the next day to use as a feature for the current day's trading
            df['prev_day_volatility'] = df['daily_volatility'].shift(1)
            
            ### Pricing Features 

            # Reference to WAP Ratio
            df['RefWAPRatio'] = df['reference_price'] / df['wap']

            # Calculate the closing price of the previous day
            df['prev_day_close'] = df.groupby('stock_id')['wap'].shift(1)

            # Calculate the opening price of the current day (assuming the first entry of the day is the opening price)
            df['current_day_open'] = df.groupby(['stock_id', 'date_id'])['wap'].transform('first')

            # Calculate the overnight return
            df['overnight_return'] = (df['current_day_open'] - df['prev_day_close']) / df['prev_day_close']

           
            # Handling NaN and infinite values
            
            #df.fillna(0.0, inplace=True) #test and see
            
            #df = df.to_pandas()

            grouped = df.groupby(['stock_id', 'date_id'])

            # Define the transformations for each set of features
            price_features_diff = ['imbalance_size', 'reference_price', 'bid_price', 'bid_size', 'ask_price', 'ask_size', 'near_price', 'far_price', 'wap']
            price_features_roc = ['imbalance_size', 'reference_price', 'matched_size', 'bid_price', 'bid_size', 'ask_price', 'ask_size', 'far_price', 'wap']
            price_features_ema = ['bid_size', 'ask_size']
            price_features_ema_diff = ['reference_price', 'matched_size', 'bid_price', 'ask_price', 'wap']
            price_features_momentum = ['reference_price', 'matched_size', 'bid_size', 'ask_price', 'wap']

            # Apply transformations to their corresponding features
            for feature in price_features_diff:
                df[f'{feature}_diff'] = grouped[feature].transform(lambda x: x.diff())

            for feature in price_features_roc:
                df[f'{feature}_roc'] = grouped[feature].transform(lambda x: x.pct_change())

            ema_span = 6  # You can adjust this value as needed
            for feature in price_features_ema:
                df[f'{feature}_ema'] = grouped[feature].transform(lambda x: x.ewm(span=ema_span, adjust=False).mean())

            for feature in price_features_ema_diff:
                df[f'{feature}_ema_diff'] = df[feature] - grouped[feature].transform(lambda x: x.ewm(span=ema_span, adjust=False).mean())

            momentum_period = 3  # You can adjust this value as needed
            for feature in price_features_momentum:
                df[f'{feature}_momentum'] = grouped[feature].transform(lambda x: x.diff(periods=momentum_period))

            ### Technical Indicators 
            # Group the DataFrame by 'stock_id' and 'date_id'
            grouped = df.groupby(['stock_id','date_id'])

            # MACD
            df['ema9'] = grouped['wap'].transform(lambda x: x.ewm(span=9, adjust=False).mean())
            df['ema26'] = grouped['wap'].transform(lambda x: x.ewm(span=26, adjust=False).mean())
            df['macd'] = df['ema9'] - df['ema26']
            df['signal'] = grouped['macd'].transform(lambda x: x.ewm(span=9, adjust=False).mean())
            df['histogram'] = df['macd'] - df['signal']

            # RSI
            delta = grouped['wap'].transform(lambda x: x.diff())
            gain = delta.where(delta > 0, 0).fillna(0)
            loss = -delta.where(delta < 0, 0).fillna(0)
            avg_gain = gain.transform(lambda x: x.rolling(window=14).mean())
            avg_loss = loss.transform(lambda x: x.rolling(window=14).mean())
            rs = avg_gain / avg_loss
            df['RSI'] = 100 - (100 / (1 + rs))

            # Other Indicators
            df['RoC13'] = grouped['wap'].transform(lambda x: x.pct_change(periods=13)) * 100
            df['MOM12'] = grouped['wap'].transform(lambda x: x.diff(12))

            # Stochastic Oscillator (K15 and D5)
            low_min = grouped['wap'].transform(lambda x: x.rolling(window=15).min())
            high_max = grouped['wap'].transform(lambda x: x.rolling(window=15).max())
            df['K15'] = 100 * ((df['wap'] - low_min) / (high_max - low_min))
            df['D5'] = df['K15'].rolling(window=5).mean()

            # SMA20C
            df['SMA20C'] = grouped['wap'].transform(lambda x: x.rolling(window=20).mean())

            # Bollinger Bands
            df['BOLlow'] = df['SMA20C'] - 2 * grouped['wap'].transform(lambda x: x.rolling(window=20).std())
            df['BOLup'] = df['SMA20C'] + 2 * grouped['wap'].transform(lambda x: x.rolling(window=20).std())
            df['BOL'] = 100 * (df['wap'] - df['BOLlow']) / (df['BOLup'] - df['BOLlow'])


            df.fillna(method='bfill', inplace=True)

   

            #df = cudf.from_pandas(df)
            # Lagged Features
            features = ['imbalance_size', 'reference_price', 'matched_size', 'bid_size', 'ask_size','target']

            for feature in features:  
                df[f'{feature}_lag_10s'] = df.groupby(['stock_id','date_id'])[feature].shift(1).fillna(0)
            
            
            #df=df.to_pandas()
            

            # Apply rolling calculations for 'target'
            window_sizes_target_sec = [2, 18]  # Window sizes for 'target'
            window_sec=[20,180] 
            for window,sec in zip(window_sizes_target_sec,window_sec):
                mean_col_name = f'target_rm_{sec}s'
                std_col_name = f'target_rs_{sec}s'
                df[mean_col_name], df[std_col_name] = apply_rolling(df, 'target', window)

            
                      

            #df=df.to_pandas()
            df.replace([np.inf, -np.inf], 0, inplace=True)             
            df.fillna(method='bfill', inplace=True)
            

            return df

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-12-20T21:13:12.190906Z","iopub.execute_input":"2023-12-20T21:13:12.191328Z","iopub.status.idle":"2023-12-20T21:13:12.251723Z","shell.execute_reply.started":"2023-12-20T21:13:12.191294Z","shell.execute_reply":"2023-12-20T21:13:12.249886Z"}}

@njit(fastmath=True, parallel=True)
def rolling_mean(arr, window, min_periods):
    n = len(arr)
    means = np.empty(n, dtype=np.float64)
    stds = np.empty(n, dtype=np.float64)

    for i in prange(n):
        if i < window:
            sub_arr = arr[max(0, i - min_periods + 1):i + 1]
        else:
            sub_arr = arr[i - window + 1:i + 1]
        
        if len(sub_arr) >= min_periods:
            means[i] = np.mean(sub_arr)
            stds[i] = np.nan if len(sub_arr) < 2 else np.std(sub_arr)
        else:
            means[i] = np.nan
            stds[i] = np.nan

    return means, stds

@njit(fastmath=True, parallel=True)
def rolling_mean_target(arr, window, min_periods):
    n = len(arr)
    means = np.empty(n, dtype=np.float64)
    stds = np.empty(n, dtype=np.float64)

    for i in prange(n):
        if i < window:
            # Use data up to but not including the current index
            sub_arr = arr[max(0, i - min_periods + 1):i]
        else:
            # Exclude the current point from the window
            sub_arr = arr[i - window:i]
        
        if len(sub_arr) >= min_periods:
            means[i] = np.mean(sub_arr)
            # Calculate standard deviation, handle cases with less than two elements
            stds[i] = np.std(sub_arr) if len(sub_arr) > 1 else np.nan
        else:
            means[i] = np.nan
            stds[i] = np.nan

    return means, stds



def apply_rolling(df, feature, window_size, min_periods=1):
    means, stds = np.empty(len(df)), np.empty(len(df))
    grouped = df.groupby(['stock_id'])[feature]
    if feature != 'target':
        for group_key, group_values in grouped:
            m, s = rolling_mean(group_values.values, window_size, min_periods)
            means[group_values.index] = m
            stds[group_values.index] = s
    
    elif feature == 'target':
        for group_key, group_values in grouped:
            m,s= rolling_mean_target(group_values.values, window_size, min_periods)
            means[group_values.index] = m
            stds[group_values.index] = s
            
    return means, stds



        
def feature_engineering_test(test,train):

            test.sort_values(by=['stock_id','date_id','seconds_in_bucket'], inplace=True)
            

            # Define the columns that you want to check and interpolate
            cols_to_interpolate = ['imbalance_size', 'reference_price', 'matched_size', 'far_price', 'near_price', 
                        'bid_price', 'bid_size', 'ask_price', 'ask_size', 'wap', 'imbalance_buy_sell_flag']

            # Check for NaN values and interpolate only if NaNs are present
            for col in cols_to_interpolate:
                if test[col].isna().any():
                    test[col] = test[col].interpolate(method='cubic')

            # Load the 'model_far' model
            with open('/kaggle/input/xgboostmodelling/modelfar.pkl', 'rb') as file:
                model_far = pickle.load(file)
            # Check if 'far_price' column has any NaN values
            if test['far_price'].isna().any():
                # Identify the rows where 'far_price' is NaN
                nan_far_index = test[test['far_price'].isna()].index
                # Make predictions only for NaN values
                test.loc[nan_far_index, 'far_price'] = model_far.predict(test.loc[nan_far_index, ['imbalance_buy_sell_flag', 'reference_price']])
            
            
            # Load the 'model_near' model
            with open('/kaggle/input/xgboostmodelling/modelnear.pkl', 'rb') as file:
                model_near = pickle.load(file)
            # Check if 'near_price' column has any NaN values
            if test['near_price'].isna().any():
                # Identify the rows where 'near_price' is NaN
                nan_near_index = test[test['near_price'].isna()].index
                # Make predictions only for NaN values
                test.loc[nan_near_index, 'near_price'] = model_near.predict(test.loc[nan_near_index, ['bid_price', 'ask_price', 'wap', 'reference_price', 'imbalance_buy_sell_flag']])
                
            #test = cudf.from_pandas(test)
            ######################
            # Imbalance Features #
            ######################

            # Imbalance Direction
            test['ImbalanceDirection'] = test['imbalance_size'] * test['imbalance_buy_sell_flag']

            # Book Imbalance
            test['BookImbalance'] = (test['bid_size'] - test['ask_size']) / (test['bid_size'] + test['ask_size'])


            # Price Imbalance
            test['PriceImbalance'] = test['reference_price'] - test['wap']


            
            
            ######################################
            # Order Book, Order Flow, and Volume #
            ######################################

            ### Order Book Features
            # Bid-Ask Spread
            test['BidAskSpread'] = test['ask_price'] - test['bid_price']

            # MidPrice and its Movement
            test['MidPrice'] = (test['ask_price'] + test['bid_price']) / 2
            test['MidPrice_Movement'] = test['MidPrice'].diff().fillna(0)
            
            # Price Spread
            test['PriceSpread'] = test['ask_price'] - test['bid_price']

            # Normalized Spread
            test['NormSpread'] = test['PriceSpread'] / test['wap']


            # Size Ratio
            test['SizeRatio'] = test['bid_size'] / test['ask_size']

            # Volumed and volume
            test['volume'] = test['matched_size'].pct_change() * 100

            ### Pricing Features 

            # Reference to WAP Ratio
            test['RefWAPRatio'] = test['reference_price'] / test['wap']
            
            # Calculate the closing price of the previous day
            test['prev_day_close'] = test.groupby('stock_id')['wap'].shift(1)

            # Calculate the opening price of the current day (assuming the first entry of the day is the opening price)
            test['current_day_open'] = test.groupby(['stock_id', 'date_id'])['wap'].transform('first')

            # Calculate the overnight return
            test['overnight_return'] = (test['current_day_open'] - test['prev_day_close']) / test['prev_day_close']



            # Handling NaN and infinite values
            #test.replace([np.inf, -np.inf], np.nan, inplace=True)
            #test.fillna(0, inplace=True)
            #test = test.to_pandas()
        
            
            concatenated_data = pd.concat([train,test], ignore_index=True)
            
            concatenated_data.sort_values(by=['stock_id','date_id','seconds_in_bucket'], inplace=True)
            
            ### Order Flow Features
            
            
             
            # Split DataFrame into groups
 
            grouped = concatenated_data.groupby(['stock_id','date_id'])

            # Define the transformations for each set of features
            price_features_diff = ['imbalance_size', 'reference_price', 'bid_price', 'bid_size', 'ask_price', 'ask_size', 'near_price', 'far_price', 'wap']
            price_features_roc = ['imbalance_size', 'reference_price', 'matched_size', 'bid_price', 'bid_size', 'ask_price', 'ask_size', 'far_price', 'wap']
            price_features_ema = ['bid_size', 'ask_size']
            price_features_ema_diff = ['reference_price', 'matched_size', 'bid_price', 'ask_price', 'wap']
            price_features_momentum = ['reference_price', 'matched_size', 'bid_size', 'ask_price', 'wap']

            # Apply transformations to their corresponding features
            for feature in price_features_diff:
                concatenated_data[f'{feature}_diff'] = grouped[feature].transform(lambda x: x.diff())

            for feature in price_features_roc:
                concatenated_data[f'{feature}_roc'] = grouped[feature].transform(lambda x: x.pct_change())

            ema_span = 6  # You can adjust this value as needed
            for feature in price_features_ema:
                concatenated_data[f'{feature}_ema'] = grouped[feature].transform(lambda x: x.ewm(span=ema_span, adjust=False).mean())

            for feature in price_features_ema_diff:
                concatenated_data[f'{feature}_ema_diff'] = concatenated_data[feature] - grouped[feature].transform(lambda x: x.ewm(span=ema_span, adjust=False).mean())

            momentum_period = 3  # You can adjust this value as needed
            for feature in price_features_momentum:
                concatenated_data[f'{feature}_momentum'] = grouped[feature].transform(lambda x: x.diff(periods=momentum_period))

                
            grouped = concatenated_data.groupby(['stock_id','date_id'])
            
            # MACD
            concatenated_data['ema9'] = grouped['wap'].transform(lambda x: x.ewm(span=9, adjust=False).mean())
            concatenated_data['ema26'] = grouped['wap'].transform(lambda x: x.ewm(span=26, adjust=False).mean())
            concatenated_data['macd'] = concatenated_data['ema9'] - concatenated_data['ema26']
            concatenated_data['signal'] = grouped['macd'].transform(lambda x: x.ewm(span=9, adjust=False).mean())
            concatenated_data['histogram'] = concatenated_data['macd'] - concatenated_data['signal']

            # RSI
            delta = grouped['wap'].transform(lambda x: x.diff())
            gain = delta.where(delta > 0, 0).fillna(0)
            loss = -delta.where(delta < 0, 0).fillna(0)
            avg_gain = gain.transform(lambda x: x.rolling(window=14).mean())
            avg_loss = loss.transform(lambda x: x.rolling(window=14).mean())
            rs = avg_gain / avg_loss
            concatenated_data['RSI'] = 100 - (100 / (1 + rs))

            # Other Indicators
            concatenated_data['RoC13'] = grouped['wap'].transform(lambda x: x.pct_change(periods=13)) * 100
            concatenated_data['MOM12'] = grouped['wap'].transform(lambda x: x.diff(12))

            # Stochastic Oscillator (K15 and D5)
            low_min = grouped['wap'].transform(lambda x: x.rolling(window=15).min())
            high_max = grouped['wap'].transform(lambda x: x.rolling(window=15).max())
            concatenated_data['K15'] = 100 * ((concatenated_data['wap'] - low_min) / (high_max - low_min))
            concatenated_data['D5'] = concatenated_data['K15'].rolling(window=5).mean()

            # SMA20C
            concatenated_data['SMA20C'] = grouped['wap'].transform(lambda x: x.rolling(window=20).mean())

            # Bollinger Bands
            concatenated_data['BOLlow'] = concatenated_data['SMA20C'] - 2 * grouped['wap'].transform(lambda x: x.rolling(window=20).std())
            concatenated_data['BOLup'] = concatenated_data['SMA20C'] + 2 * grouped['wap'].transform(lambda x: x.rolling(window=20).std())
            concatenated_data['BOL'] = 100 * (concatenated_data['wap'] - concatenated_data['BOLlow']) / (concatenated_data['BOLup'] - concatenated_data['BOLlow'])


            concatenated_data.fillna(method='bfill', inplace=True)


 
            ###############################
            # lagged and rolling features #
            ###############################
            # Lagged Features
            features = ['imbalance_size', 'reference_price', 'matched_size', 'bid_size', 'ask_size','target']

            for feature in features:  
                concatenated_data[f'{feature}_lag_10s'] = concatenated_data.groupby(['stock_id','date_id'])[feature].shift(1).fillna(0)
            
            #concatenated_data=concatenated_data.to_pandas()
            
            
  
            
            # Apply rolling calculations for 'target'
            window_sizes_target_sec = [2,18]  # Window sizes for 'target'
            window_sec=[20,180] 
            for window,sec in zip(window_sizes_target_sec,window_sec):
                mean_col_name = f'target_rm_{sec}s'
                std_col_name = f'target_rs_{sec}s'
                concatenated_data[mean_col_name], concatenated_data[std_col_name] = apply_rolling(concatenated_data, 'target', window)

            
                      
            
            #df=df.to_pandas()
            concatenated_data.replace([np.inf, -np.inf], 0, inplace=True)             
            concatenated_data.fillna(method='bfill', inplace=True)
            
            
                      



            return concatenated_data

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-12-20T21:13:12.257321Z","iopub.execute_input":"2023-12-20T21:13:12.257823Z","iopub.status.idle":"2023-12-20T21:13:12.267968Z","shell.execute_reply.started":"2023-12-20T21:13:12.257781Z","shell.execute_reply":"2023-12-20T21:13:12.266957Z"}}
def load_data(train, test):
    # Append 'target' and 'time_id' columns to test if they don't exist
    if 'target' not in test.columns:
        test['target'] = 0
    if 'time_id' not in test.columns:
        test['time_id'] = -999

    # List of original columns to exclude from NA filling
    original_columns = ['imbalance_size', 'reference_price', 'matched_size', 'far_price', 'near_price', 
                        'bid_price','bid_size', 'ask_price', 'ask_size', 'wap', 'imbalance_buy_sell_flag'
                        ,'stock_id', 'date_id', 'row_id', 'time_id', 'target','seconds_in_bucket']
    
    # Determine which columns are in train but not in test, excluding the original columns
    columns_to_fill_na = set(train.columns) - set(test.columns) - set(original_columns)

    # Fill these columns with NA in the test data
    for column in columns_to_fill_na:
        test[column] = np.nan

    # Ensure the order of columns in test matches that of train
    test = test[train.columns]
    # Process test data through feature engineering
    test_processed = feature_engineering_test(test,train)

    return test_processed

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-12-20T21:13:12.269188Z","iopub.execute_input":"2023-12-20T21:13:12.269538Z"}}
# Get training data 
train_data = pd.read_csv('/kaggle/input/optiver-trading-at-the-close/train.csv')
train_data=reduce_mem_usage(train_data)
train_data = train_data[(train_data['date_id'].between(0, 477))]
# Process the training data once to calculate all necessary features
train_data_processed = feature_engineering_train(train_data)

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Process the training data once to calculate all necessary features
train_data_processed = train_data_processed.query('470 <= date_id <= 477')
last_time_id = train_data_processed['time_id'].max()

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Define your mock API loop here

Modelling='Na'
# Define your mock API loop here
for (test, rev, sample_prediction) in iter_test:
    columns_given = ['seconds_in_bucket', 'imbalance_size',
                     'imbalance_buy_sell_flag', 'reference_price', 'matched_size',
                     'far_price', 'near_price', 'bid_price', 'bid_size',
                     'ask_price', 'ask_size', 'wap', 'currently_scored']
    test[columns_given] = test[columns_given].astype(float)
    
    # Process test data
    test_processed = load_data(train_data_processed, test, stock_stats)
    test_processed = test_processed[test_processed['time_id'] == -999]
    test_processed = test_processed.loc[:, ~test_processed.columns.duplicated()]
    
    # Check currently_scored flag and if Modelling is set to 'Neural'
    if test.iloc[0]['currently_scored'] == False:
        if Modelling == 'Neural':
            processed_test_data = proccess_data_static(test, Stock_Stats)

            # Load the scaler model from disk
            with open('CNNBILSTMATN_Static_Scaling.pkl', 'rb') as file:
                scaler = pickle.load(file)

            # Define the feature columns to scale
            feature_cols = [col for col in processed_test_data.columns if col not in ['row_id', 'stock_id', 'date_id', 'time_id', 'seconds_in_bucket','target','imbalance_buy_sell_flag','Unnamed: 0','currently_scored']]
            df_scaled = scaler.transform(processed_test_data[feature_cols])
            processed_test_data[feature_cols] = df_scaled

            # Load Neural model and make predictions
            Neural_model.gpu_id = 0
            Neural_model.predictor = "auto"
            feature_cols = [col for col in processed_test_data.columns if col not in ['row_id', 'stock_id', 'date_id', 'time_id','target','Unnamed: 0','currently_scored']]

            stock_predictions = []
            for stock_id in range(200):
                stock_data = processed_test_data[processed_test_data['stock_id'] == stock_id]
                 # Pad the data to ensure it has 54 rows
                while len(stock_data) < 54:
                    # Duplicate the last available row to pad the data
                    last_row = stock_data.iloc[-1:].copy()
                    stock_data = pd.concat([stock_data, last_row], ignore_index=True)

                # Extract the input data for the current row
                input_data = stock_data[feature_cols].values

                # Reshape the input data to match the model's input shape
                input_data = input_data.astype('float32')
                
            # Convert the input_data into a NumPy array
                input_data = np.array(input_data)

                # Calculate the number of rows in the input data
                num_rows = input_data.shape[0]

                # Calculate the number of times to repeat the data along axis 0 to match 54 rows
                repeat_times = np.ceil(54 / num_rows).astype(int)

                # Repeat the data along axis 0
                reshaped_data = np.repeat(input_data, repeat_times, axis=0)

                # Trim or pad the data to have exactly 54 rows
                reshaped_data = reshaped_data[:54]

                # Reshape the data to the desired shape (None, 54, 36)
                reshaped_data = reshaped_data.reshape(-1, 54, 36)
            
                predictions = Neural_model.predict(reshaped_data)
                stock_predictions.append(predictions)

            sample_prediction['target'] = np.concatenate(stock_predictions)
            env.predict(sample_prediction)

        else:
            # Prepare features for non-Neural model prediction
            feature_names = model.get_booster().feature_names
            test_predict = test_predict.reindex(columns=feature_names)

            model.gpu_id = 0
            model.predictor = "auto"
            predictions = model.predict(test_predict)
            sample_prediction['target'] = predictions
            env.predict(sample_prediction)

        # Update train_data_processed if predictions were made
        if test.iloc[0]['currently_scored'] == False:
                test_processed['time_id'] = [last_time_id + 1] * 200
                test_processed['target'] = predictions
                train_data_processed = pd.concat([train_data_processed, test_processed])
                last_time_id = train_data_processed['time_id'].max()

            # Keep only data for the last two days
        unique_dates = train_data_processed['date_id'].unique()
        if len(unique_dates) > 1:
                last_two_dates = sorted(unique_dates)[-2:]
                train_data_processed = train_data_processed[train_data_processed['date_id'].isin(last_two_dates)]

        # Sort data
        train_data_processed.sort_values(by=['stock_id', 'date_id', 'seconds_in_bucket'], inplace=True)
        
    else:
        # Provide dummy predictions for unscored rows
        sample_prediction['target'] = 0 # or any default value
