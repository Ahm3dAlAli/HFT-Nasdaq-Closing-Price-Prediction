# %% [markdown]
# # Optivar Trading at the Close | Modelling, Training and Tuning
# 
# ## Introduction
# 
# Welcome to the Optivar Trading at the Close project. The goal of this notebook is to provide a comprehensive guide to designing, training, and evaluating machine learning models for predicting stock price movements at market close (HFT). Trading at the close is a significant part of daily market activity, and understanding the predictive patterns can offer a competitive edge.
# 
# ### Structure of the Notebook
# 
# - **Data Preprocessing**: The data will be loaded and partitioned into training and testing sets based on given `date_id` values. Features and target variables will be separated, ensuring that identifiers such as `row_id`, `target`, `date_id`, and `time_id` are excluded from the feature set.
# 
# - **Model Initialization and Training**: We'll use XGBoost as our primary regression model, with parameters specifically tuned for our use-case. Early stopping rounds are used to prevent overfitting.
# 
# - **Feature Importance**: Once the model is trained, we'll look at feature importances to understand what variables are most influential in our predictions. 
# 
# - **Model Evaluation**: The performance of the model will be evaluated on the test set, specifically using the Mean Absolute Error (MAE) as our metric.
# 
# - **Bayesian Hyperparameter Tuning**: The model is ran hrough set of paremters ro be optimized to get the most suitable ones.
# 
# - **Deep Learning with Attention Mechanism**: We'll employ a specialized deep learning architecture that makes use of Convolutional and Bidirectional LSTM layers, combined with an attention mechanism to weigh the importance of different time steps in the sequence.
# 
# - **Model Training on TPU**: To speed up the training process, we'll utilize Tensor Processing Units (TPUs), providing an efficient way to handle large datasets and complex models.
# 
# - **Model Serialization**: Finally, the trained model will be saved for future use or deployment.
# 
# ### Objective
# 
# Our main objective is to build a robust predictive model that can accurately forecast stock price movements at the close of trading. The model should be flexible enough to adapt to new data, scalable to handle large datasets, and fast enough to provide real-time predictions.
# 
# Let's dive in!

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-12-20T19:28:56.137560Z","iopub.execute_input":"2023-12-20T19:28:56.137895Z","iopub.status.idle":"2023-12-20T19:29:20.983593Z","shell.execute_reply.started":"2023-12-20T19:28:56.137867Z","shell.execute_reply":"2023-12-20T19:29:20.982522Z"}}
%pip install --upgrade pip

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-12-20T19:29:20.986213Z","iopub.execute_input":"2023-12-20T19:29:20.986603Z","iopub.status.idle":"2023-12-20T19:29:32.280087Z","shell.execute_reply.started":"2023-12-20T19:29:20.986569Z","shell.execute_reply":"2023-12-20T19:29:32.279040Z"}}
%pip install xgboost

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-12-20T19:29:32.281501Z","iopub.execute_input":"2023-12-20T19:29:32.281794Z","iopub.status.idle":"2023-12-20T19:29:43.757310Z","shell.execute_reply.started":"2023-12-20T19:29:32.281768Z","shell.execute_reply":"2023-12-20T19:29:43.756195Z"}}
%pip install bayesian-optimization

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-12-20T19:29:43.758824Z","iopub.execute_input":"2023-12-20T19:29:43.759131Z","iopub.status.idle":"2023-12-20T19:30:10.430904Z","shell.execute_reply.started":"2023-12-20T19:29:43.759103Z","shell.execute_reply":"2023-12-20T19:30:10.430065Z"}}
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

#Plotting
import matplotlib.pyplot as plt


#Hyperparemter tuning 
from bayes_opt import BayesianOptimization

#Speed Processing and Parallel computations

import multiprocessing
import gc  
import os  
import time  
import warnings  
from itertools import combinations  
from warnings import simplefilter  
import logging
from numba import njit, prange
#import cudf

# Disable warnings to keep the code clean
warnings.filterwarnings("ignore")
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

data=pd.read_csv("/kaggle/input/optiver-trading-at-the-close/train.csv")

# %% [markdown]
# ==============================================================================

# %% [markdown]
# ==============================================================================

# %% [markdown]
# # Reducing size of data

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-12-20T19:30:10.433401Z","iopub.execute_input":"2023-12-20T19:30:10.433684Z","iopub.status.idle":"2023-12-20T19:30:11.248089Z","shell.execute_reply.started":"2023-12-20T19:30:10.433658Z","shell.execute_reply":"2023-12-20T19:30:11.247056Z"}}
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

df=reduce_mem_usage(data, verbose=True)

# %% [markdown]
# ==============================================================================

# %% [markdown]
# ==============================================================================

# %% [markdown]
# # Feature Engineering 
# 
# ## Overview
# The feature engineering functions manipulates a DataFrame (`df`) which contains time-series financial data. The data relates to bid-ask prices, order size, etc., for different stocks at various time intervals. The purpose of feature engineering is to create new features that can improve the performance of predictive models. Now different Engineering is applied depdening on the requiremnts of testing and the capiability to run large models
# 
# ---
# 
# ## Steps in Feature Engineering:
# 
# ### Data Preprocessing
# 1. **Remove Data with No Target Value**: Deletes rows where the target column is null.
# 
# ---
# 
# ### Data Imputation
# 1. **Linear Interpolation**: Used for filling in missing data points in certain columns.
# 2. **Model-Based Imputation**: Linear regression models are trained to predict missing values for 'far_price' and 'near_price'.
# 
# ---
# 
# ### Outlier Handling
# 1. **Z-score Normalization**: Removes outliers based on z-score.
# 
# ---
# 
# ### Derived Features
# 1. **Imbalance Features**: Calculate imbalance direction, book imbalance, price imbalance, and WAP imbalance.
# 2. **Order Book, Order Flow, and Volume Features**: Include bid-ask spread, mid-price and its movement, normalized spread, etc.
# 3. **Pricing Features**: Includes crossing delta and a reference to WAP ratio.
# 4. **Technical Indicators**: Include MACD, RSI, RoC13, etc.
# 
# ---
# 
# ### Statistical Features
# 1. **Open and Close Prices**: For each stock and date.
# 
# ---
# 
# ### Lagged and Rolling Features
# 1. **Lagged Features**: Past values (lags) for various features are added.
# 2. **Rolling Means and Standard Deviations**: These are calculated for the target variable and some feature variables, either based on a day or on a 90-second window.
# 
# ---
# 
# The function concludes by filling in any remaining NaN values and returns the modified DataFrame.

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-12-20T19:30:11.249459Z","iopub.execute_input":"2023-12-20T19:30:11.249723Z","iopub.status.idle":"2023-12-20T19:50:17.934396Z","shell.execute_reply.started":"2023-12-20T19:30:11.249699Z","shell.execute_reply":"2023-12-20T19:50:17.933553Z"}}

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
    grouped = df.groupby('stock_id')[feature]

    for group_key, group_values in grouped:
        group_values = group_values.reset_index(drop=True)  # Reset the index for continuity
        if feature != 'target':
            m, s = rolling_mean(group_values.values, window_size, min_periods)
        else:
            m, s = rolling_mean_target(group_values.values, window_size, min_periods)
        
        # Ensure the indices are within bounds
        group_indices = group_values.index.to_numpy()
        valid_indices = group_indices[group_indices < len(means)]
        means[valid_indices] = m[:len(valid_indices)]
        stds[valid_indices] = s[:len(valid_indices)]
    
    return means, stds




    
def feature_engineering_dynamic(df):
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
            grouped = df.groupby(['stock_id','date_id'])

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
            window_sizes_target_sec = [2,18]  # Window sizes for 'target'
            window_sec=[20,180] 
            for window,sec in zip(window_sizes_target_sec,window_sec):
                mean_col_name = f'target_rm_{sec}s'
                std_col_name = f'target_rs_{sec}s'
                df[mean_col_name], df[std_col_name] = apply_rolling(df, 'target', window)

                      

            #df=df.to_pandas()
            df.replace([np.inf, -np.inf], 0, inplace=True)             
            df.fillna(method='bfill', inplace=True)
            

            return df


# Preporccess and Extract Features 
df_dynamic= feature_engineering_dynamic(df)

# %% [markdown]
# ==============================================================================

# %% [markdown]
# ==============================================================================

# %% [markdown]
# # XGBoost Regression Model
# 
# In this section, we demonstrate how to build and evaluate an XGBoost Regression model. We will be using the dataset `df`, which is assumed to be pre-loaded and having the necessary columns.To train a gradient boosting model for regression, we use the `xgb.XGBRegressor` class from the XGBoost library. Here's how the model is initialized:
# 
# - learning_rate: The step size shrinkage used in the update to prevent overfitting. A value of 0.023 is used.
# 
# - max_depth: Maximum depth of the decision trees. Set to 7 in this case.
# 
# - min_child_weight: Minimum sum of instance weight needed in a child. Set to 8.
# 
# - colsample_bytree: Fraction of features to choose for each boosting round. Set to 0.85.
# 
# - subsample: Fraction of training instances to be randomly sampled in each boosting round. Set to 0.9.
# 
# - random_state: Random seed used for reproducibility. Set to 42.
# 
# - objective: Specifies the learning task and the corresponding learning objective. Here, it is set to 'reg:absoluteerror' for regression with absolute error.
# 
# - eval_metric: Evaluation metric for validation data. Mean Absolute Error (MAE) is used.
# 
# - tree_method: The tree construction algorithm used in XGBoost. Set to 'hist' for histogram-based training.

# %% [markdown]
# # Dynamic Features Modelling

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-12-20T19:50:17.935841Z","iopub.execute_input":"2023-12-20T19:50:17.936229Z","iopub.status.idle":"2023-12-20T19:50:21.031463Z","shell.execute_reply.started":"2023-12-20T19:50:17.936192Z","shell.execute_reply":"2023-12-20T19:50:21.030462Z"}}

# Define identifiers and features
identifiers = [ 'stock_id','row_id', 'target', 'date_id', 'time_id','seconds_in_bucket']


# Assume X and y are your full dataset and labels
# Define the ranges for training, validation, and test sets
train_end = 470
val_end = 477

# Create training, validation, and test sets
X_train = df_dynamic[df_dynamic['date_id'] <= train_end].drop(columns=identifiers)
y_train = df_dynamic[df_dynamic['date_id'] <= train_end]['target']

X_val = df_dynamic[(df_dynamic['date_id'] > train_end) & (df_dynamic['date_id'] <= val_end)].drop(columns=identifiers)
y_val = df_dynamic[(df_dynamic['date_id'] > train_end) & (df_dynamic['date_id'] <= val_end)]['target']

X_test = df_dynamic[df_dynamic['date_id'] > val_end].drop(columns=identifiers)
y_test = df_dynamic[df_dynamic['date_id'] > val_end]['target']

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-12-20T19:50:21.032672Z","iopub.execute_input":"2023-12-20T19:50:21.032949Z","iopub.status.idle":"2023-12-20T19:52:26.191235Z","shell.execute_reply.started":"2023-12-20T19:50:21.032925Z","shell.execute_reply":"2023-12-20T19:52:26.190205Z"}}

# Initialize model 
model_dynamic = xgb.XGBRegressor(
    learning_rate=0.023,
    gamma=0.03103,
    max_depth=7, 
    min_child_weight=8, 
    colsample_bytree=0.85,
    subsample=0.9,
    random_state=42,
    objective='reg:absoluteerror',
    eval_metric='mae',
    tree_method='hist',
    n_estimators=500
)

# Fit the model on training data and validate on validation data
model_dynamic.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=True, early_stopping_rounds=10 )

# Make predictions and evaluate on the test set
y_pred = model_dynamic.predict(df_dynamic[(df_dynamic['date_id'].between(478, 480))].drop(columns=identifiers))
test_mae = mean_absolute_error(df_dynamic[(df_dynamic['date_id'].between(478, 480))]['target'], y_pred)
print(f"Test MAE: {test_mae}")

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-12-20T20:07:06.566419Z","iopub.execute_input":"2023-12-20T20:07:06.566818Z","iopub.status.idle":"2023-12-20T20:07:06.576013Z","shell.execute_reply.started":"2023-12-20T20:07:06.566787Z","shell.execute_reply":"2023-12-20T20:07:06.575170Z"}}
with open('XGB_dynamicmodel.pkl', 'wb') as f:
    pickle.dump(model_dynamic, f)

# %% [markdown]
# ==============================================================================

# %% [markdown]
# # Hyper-Parameter tuning using Bayesian Optimization
# 
# Bayesian Optimization is used to find the optimal hyperparameters for an XGBoost Regressor model. The aim is to optimize the model's performance on a validation set, specifically by maximizing the negative Mean Absolute Error (MAE).
# 
# - Parameter Sets: Predefined sets of values are established for hyperparameters such as learning_rate, gamma, max_depth, min_child_weight, colsample_bytree, and subsample.
# 
# - Parameter Snapping: Due to the continuous nature of Bayesian Optimization, a helper function snap_to_set is implemented. This function takes a value and maps it to the nearest predefined value in the parameter sets.
# 
# - Objective Function: The function xgb_eval serves as the objective for the optimization. It:
# 
# 1. Converts continuous parameters to their nearest discrete values.
# 2. Trains the XGBoost model with the selected parameters.
# 3. Returns the negative MAE on the validation data as the objective to be maximized.
# 4. Optimization Bounds: For each hyperparameter, lower and upper bounds are set based on the predefined parameter sets.
# 
# - Optimization Execution: The Bayesian Optimization runs for a total of 100 iterations, which includes 30 initial random evaluations and 70 subsequent Bayesian-informed iterations. The objective is to find the best combination of hyperparameters that yield the highest performance.

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-12-20T19:52:26.202738Z","iopub.execute_input":"2023-12-20T19:52:26.203419Z","iopub.status.idle":"2023-12-20T19:54:29.679878Z","shell.execute_reply.started":"2023-12-20T19:52:26.203390Z","shell.execute_reply":"2023-12-20T19:54:29.678119Z"}}


# Define the sets of values you want to optimize over
param_sets = {
    'learning_rate': [0.01,0.1],
    'gamma': [0.001,0.07],
    'max_depth': [6,14],
    'min_child_weight': [6,14],
    'colsample_bytree': [0.7,0.9],
    'subsample': [0.7,0.9]
}

# Function to snap a value to the nearest value in a list
def snap_to_set(value, value_set):
    return min(value_set, key=lambda x: abs(x - value))

# Define the objective function for Bayesian optimization
def xgb_eval(learning_rate,gamma, max_depth, min_child_weight, colsample_bytree, subsample):
    # Snap values to nearest in the predefined sets
    learning_rate = snap_to_set(learning_rate, param_sets['learning_rate'])
    max_depth = int(snap_to_set(max_depth, param_sets['max_depth']))
    min_child_weight = int(snap_to_set(min_child_weight, param_sets['min_child_weight']))
    colsample_bytree = snap_to_set(colsample_bytree, param_sets['colsample_bytree'])
    subsample = snap_to_set(subsample, param_sets['subsample'])
    
    model = xgb.XGBRegressor(
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        colsample_bytree=colsample_bytree,
        subsample=subsample,
        random_state=42,
        objective='reg:absoluteerror',
        eval_metric='mae',
        tree_method='hist',
        n_estimators= 400 
    )
    
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False, early_stopping_rounds=10)
    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    return -mae  # We want to maximize the negative MAE (Minimize MAE)

# Define bounds
pbounds = {
    'learning_rate': (min(param_sets['learning_rate']), max(param_sets['learning_rate'])),
    'gamma': (min(param_sets['gamma']), max(param_sets['gamma'])),
    'max_depth': (min(param_sets['max_depth']), max(param_sets['max_depth'])),
    'min_child_weight': (min(param_sets['min_child_weight']), max(param_sets['min_child_weight'])),
    'colsample_bytree': (min(param_sets['colsample_bytree']), max(param_sets['colsample_bytree'])),
    'subsample': (min(param_sets['subsample']), max(param_sets['subsample']))
}

optimizer = BayesianOptimization(f=xgb_eval, pbounds=pbounds, random_state=42, verbose=2)
optimizer.maximize(init_points=5, n_iter=3)

# Get the best parameters from Bayesian optimization
best_params = optimizer.max['params']

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-12-20T19:54:29.680835Z","iopub.status.idle":"2023-12-20T19:54:29.681178Z","shell.execute_reply.started":"2023-12-20T19:54:29.681016Z","shell.execute_reply":"2023-12-20T19:54:29.681031Z"}}

# Snap the continuous values to the nearest values in our defined sets
best_params_snapped= {
    'learning_rate': 0.06006   ,
    'gamma': 0.06103  ,
    'max_depth':8   ,
    'min_child_weight': 6,
    'colsample_bytree': 0.71,
    'subsample': 0.89,
    
}

# Train a model using the best parameters on the full training set (combining X_train and X_val)
full_train_data = pd.concat([X_train, X_val])
full_train_target = pd.concat([y_train, y_val])

model = xgb.XGBRegressor(
    **best_params_snapped,
    random_state=42,
    objective='reg:absoluteerror',
    eval_metric='mae',
    tree_method='hist',
    n_estimators=400 
)

model.fit(full_train_data, full_train_target,verbose=True)

# Make predictions on the test set
y_pred = model.predict(X_test)
test_mae = mean_absolute_error(y_test, y_pred)
print(f"Test MAE with optimized parameters: {test_mae}")

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-12-20T19:54:29.682562Z","iopub.status.idle":"2023-12-20T19:54:29.682901Z","shell.execute_reply.started":"2023-12-20T19:54:29.682736Z","shell.execute_reply":"2023-12-20T19:54:29.682752Z"}}
with open('XGBOOST_Bayesian_Tuned.pkl', 'wb') as f:
    pickle.dump(model, f)

# %% [markdown]
# ## Feature Importance and F1 Stats
# ### Dynamic Model Features

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-12-20T20:10:07.431918Z","iopub.execute_input":"2023-12-20T20:10:07.432315Z","iopub.status.idle":"2023-12-20T20:10:07.441062Z","shell.execute_reply.started":"2023-12-20T20:10:07.432261Z","shell.execute_reply":"2023-12-20T20:10:07.440142Z"}}
model_dynamic.get_booster().get_fscore()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-12-20T19:54:29.685413Z","iopub.status.idle":"2023-12-20T19:54:29.685743Z","shell.execute_reply.started":"2023-12-20T19:54:29.685581Z","shell.execute_reply":"2023-12-20T19:54:29.685596Z"}}
# Creating DataFrame from the dictionaries
f1_scores_dict = model_dynamic.get_booster().get_fscore()

# Create a DataFrame from the F1 scores dictionary
df_f1_scores = pd.DataFrame({
    "Feature": list(f1_scores_dict.keys()),
    "F1_Score": list(f1_scores_dict.values())
})

# Create a DataFrame from the feature importances
df_importances = pd.DataFrame({
    "Feature": df_dynamic[(df_dynamic['date_id'].between(0, 477))].drop(columns=identifiers).columns,
    "Importance": model_dynamic.feature_importances_
})

# Merge the two DataFrames on 'Feature'
df_features = pd.merge(df_importances, df_f1_scores, on='Feature', how='left')

# Handling NaN values if any
df_features.fillna(0, inplace=True)  # You can choose to fill NaNs with 0 or another suitable value

# Sort the DataFrame by both 'Importance' and 'F1_Score' in ascending order
df_features_sorted = df_features.sort_values(by=["F1_Score", "Importance"], ascending=[True, True])

# Number of features to drop
n_drop = 50

# Select the features to keep, excluding the lowest n rows based on the sorted values
df_important_features = df_features_sorted.iloc[n_drop:]

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-12-20T19:54:29.686958Z","iopub.status.idle":"2023-12-20T19:54:29.687304Z","shell.execute_reply.started":"2023-12-20T19:54:29.687121Z","shell.execute_reply":"2023-12-20T19:54:29.687136Z"}}
# Assuming df_important_features['Features'] contains the names of the features you want to keep
important_features = df_important_features['Feature'].tolist()

# If you also need to include the identifiers and target variable, add them back
identifiers_and_target = ['stock_id', 'row_id', 'target', 'date_id', 'time_id', 'seconds_in_bucket']
df_dynamic = df_dynamic[identifiers_and_target + important_features]

# %% [markdown]
# ==============================================================================

# %% [markdown]
# ==============================================================================

# %% [markdown]
# # CNN-BILSTM-ATN Model
# 
# For the CNN-BiLSTM-ATN model, feature scaling is performed using the MinMaxScaler from the `sklearn.preprocessing`. The architecture of the Deep CNN-BiLSTM-ATN model is built using Keras and TensorFlow.
# It consists of the following layers:
# 
# 
# - 1D Convolutional Layer
# - Batch Normalization and Dropout
# - Bidirectional LSTM Layer
# - Attention Mechanism
# - Dense Output Layer
# 
# 
# Two types of attention mechanisms are defined:
# 
# - Attention with simple softmax activation
# - Attention with additional mean computation

# %% [markdown]
# # Considerations

# %% [markdown]
# Given the architecture of the neural network, certain transformations might be less crucial.
# 
# Here's a breakdown of potentially unnecessary parts for the neural architecture:
# 
# - Linear Interpolation: Neural networks, especially deep ones, can handle a decent amount of missing data. While interpolation is generally a good step, a more straightforward method like forward-fill or backward-fill might suffice. Also, neural networks can work with sequences where some values are NaN, as long as the NaNs aren't in the target variable.
# 
# - Outlier Handling with Z-Scores: Neural networks, especially deep architectures, are generally robust to outliers. However, handling extreme outliers could still be beneficial, but it might be overkill to filter out everything beyond 2.5 standard deviations.
# 
# - Stock-level statistics: Depending on the nature of the data, the model might capture these relations inherently, especially with attention mechanisms in place.
# 
# - Time Decay Factor: This can be useful for models like linear regression or tree-based algorithms. For neural networks, especially recurrent ones, the time decay is inherently understood because of the sequence nature of the data.
# 
# - Lagged Features: Neural networks, especially RNNs and LSTMs, have a memory mechanism that captures the temporal dependencies in the data. Therefore, creating a lot of lagged features might introduce unnecessary redundancy.
# 
# - Rolling means and standard deviations: Similar to lagged features, recurrent layers in the model would inherently capture these patterns. Thus, these rolling features might be redundant.

# %% [markdown]
# ## What to do next:
# 
# Consider training two models: one with all the features and another with potentially redundant features removed.
# Compare the performance and training time of both. If the reduced feature set model performs nearly as well or better and trains faster, it might be more beneficial to use that.

# %% [markdown]
# ## Training the Model
# 
# Training is executed with the Adam optimizer, a learning rate of 0.001, and a batch size that is adjusted for TPU execution:
# 
# 
# - optimizer = Adam(learning_rate=0.001)
# - BATCH_SIZE = 64 * tpu_strategy.num_replicas_in_sync
# 
# 
# The trained model is then saved to disk using `pickle`. Early stopping and model checkpointing mechanisms can be added through Keras callbacks, although they are not present in the given code. Considerations, lookback shouldnt cross boundaries of different stocks, sequence should be set of stocks some dont have 0-480 trading days data

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-12-20T19:54:29.688369Z","iopub.status.idle":"2023-12-20T19:54:29.688751Z","shell.execute_reply.started":"2023-12-20T19:54:29.688564Z","shell.execute_reply":"2023-12-20T19:54:29.688582Z"}}

# Step 1: Create a TPUClusterResolver object
tpu_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')
tf.config.experimental_connect_to_cluster(tpu_resolver)
tf.tpu.experimental.initialize_tpu_system(tpu_resolver)

# Step 2: Create a TPUStrategy
tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu_resolver)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-12-20T20:22:36.233676Z","iopub.execute_input":"2023-12-20T20:22:36.234056Z","iopub.status.idle":"2023-12-20T20:22:42.557582Z","shell.execute_reply.started":"2023-12-20T20:22:36.234027Z","shell.execute_reply":"2023-12-20T20:22:42.556770Z"}}


# Define the feature columns to scale
feature_cols = [col for col in df_dynamic.columns if col not in ['row_id', 'stock_id', 'date_id', 'time_id', 'seconds_in_bucket','target','imbalance_buy_sell_flag']]

# Initialize the scaler and fit it on the feature columns
scaler = MinMaxScaler().fit(df_dynamic[feature_cols])

# Save the pca model to disk
with open('CNNBILSTMATN_dynamic_Scaling.pkl', 'wb') as f:
    pickle.dump(scaler, f)
    
    
# Transform the feature columns
df_scaled = scaler.transform(df_dynamic[feature_cols])

# Replace original columns with scaled ones
df_dynamic[feature_cols] = df_scaled

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-12-20T20:23:25.583396Z","iopub.execute_input":"2023-12-20T20:23:25.583767Z","iopub.status.idle":"2023-12-20T20:23:25.610664Z","shell.execute_reply.started":"2023-12-20T20:23:25.583735Z","shell.execute_reply":"2023-12-20T20:23:25.609459Z"}}

with tpu_strategy.scope():
    def get_activations(model, inputs, print_shape_only=False, layer_name=None):
            # Documentation is available online on Github at the address below.
            # From: https://github.com/philipperemy/keras-visualize-activations
            print('----- activations -----')
            activations = []
            inp = model.input
            if layer_name is None:
                outputs = [layer.output for layer in model.layers]
            else:
                outputs = [layer.output for layer in model.layers if layer.name == layer_name]  # all layer outputs
            funcs = [K.function([inp] + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions
            layer_outputs = [func([inputs, 1.])[0] for func in funcs]
            for layer_activations in layer_outputs:
                activations.append(layer_activations)
                if print_shape_only:
                    print(layer_activations.shape)
                else:
                    print('shape',layer_activations.shape)
                    print(layer_activations)
            return activations

    SINGLE_ATTENTION_VECTOR = False
    def attention_3d_block(inputs, layer_name=None):
            # inputs.shape = (batch_size, time_steps, input_dim)
            input_dim = int(inputs.shape[2])
            a = inputs
            a = Dense(input_dim, activation='softmax')(a)
            if layer_name:
                a_probs = Permute((1, 2), name=layer_name)(a)
            else:
                a_probs = Permute((1, 2))(a)

            output_attention_mul = Multiply()([inputs, a_probs])
            return output_attention_mul



    # Another way of writing the attention mechanism is suitable for the use of the above error source:https://blog.csdn.net/uhauha2929/article/details/80733255
    def attention_3d_block2(inputs, single_attention_vector=False):
            # If the upper layer is LSTM, you need return_sequences=True
            # inputs.shape = (batch_size, time_steps, input_dim)
            time_steps = K.int_shape(inputs)[1]
            input_dim = K.int_shape(inputs)[2]
            a = Permute((2, 1))(inputs)
            a = Dense(time_steps, activation='softmax')(a)
            if single_attention_vector:
                a = Lambda(lambda x: K.mean(x, axis=1))(a)
                a = RepeatVector(input_dim)(a)

            a_probs = Permute((2, 1))(a)
            # Multiplied by the attention weight, but there is no summation, it seems to have little effect
            # If you classify tasks, you can do Flatten expansion
            # element-wise
            output_attention_mul = Multiply()([inputs, a_probs])
            return output_attention_mul

    def attention_model():
            # Input layer TIME_STEPS
            inputs = Input(shape=(TIME_STEPS,INPUT_DIMS))

            # First Conv1D layer
            x = Conv1D(filters=32, kernel_size=1,activation='relu')(inputs)
            x = BatchNormalization()(x)
            x = Dropout(0.3)(x)

            # Second Conv1D layer
            x2 = Conv1D(filters=16, kernel_size=1, activation='relu')(x)
            x2 = BatchNormalization()(x2)
            x2 = Dropout(0.1)(x2)

            # Bidirectional LSTM layer
            lstm_out = Bidirectional(LSTM(lstm_units, return_sequences=True))(x2)
            lstm_out = BatchNormalization()(lstm_out)
            lstm_out = Dropout(0.3)(lstm_out)

            # Second Bidirectional LSTM layer
            lstm_out2 = Bidirectional(LSTM(lstm_units // 2, return_sequences=True))(lstm_out)
            lstm_out2 = BatchNormalization()(lstm_out2)
            lstm_out2 = Dropout(0.1)(lstm_out2)

            # Attention layer
            attention_mul = attention_3d_block(lstm_out, layer_name="attention_vec1")
            attention_mul2 = attention_3d_block(lstm_out2, layer_name="attention_vec2")

            # Concatenate the two attention layers
            merged_attention = Concatenate(axis=-1)([attention_mul, attention_mul2])

            # Flatten Layer
            merged_attention = Flatten()(merged_attention)

            # Dense layer
            dense_out = Dense(64, activation='relu')(attention_mul)
            dense_out = BatchNormalization()(dense_out)

            # Output layer
            output = Dense(1, activation='linear')(dense_out)

            # Compile model
            model = Model(inputs=[inputs], outputs=output)

            return model

    def create_dataset_no(df):
        identifiers = ['stock_id', 'stock_id_', 'row_id', 'target', 'time_id', 'date_id']
        features = [col for col in df.columns if col not in identifiers]

        dataX, dataY = [], []

        # Grouping by stock_id
        grouped = df.groupby('stock_id')

        for _, group in grouped:
            group_sorted = group.sort_values(by='date_id')
            group_values = group_sorted[features].values
            group_target = group_sorted['target'].values

            for i in range(len(group_values)):
                dataX.append(group_values[i:i + 1])  # Only include the current time step
                dataY.append(group_target[i])

        return np.array(dataX), np.array(dataY)


    def create_dataset(df, look_back=1):
            identifiers = ['stock_id', 'row_id', 'target', 'time_id', 'date_id','seconds_in_bucket']
            features = [col for col in df.columns if col not in identifiers]

            dataX, dataY = [], []

            # Grouping by stock_id
            grouped = df.groupby('stock_id')

            for _, group in grouped:
                group_sorted = group.sort_values(by='date_id')
                group_values = group_sorted[features].values
                group_target = group_sorted['target'].values

                for i in range(len(group_values) - look_back):
                    dataX.append(group_values[i:i+look_back])
                    dataY.append(group_target[i+look_back])

            return np.array(dataX), np.array(dataY)



    model_path = './model.h5'
    # TRAIN
    INPUT_DIMS = 82
    TIME_STEPS = 54
    lstm_units = 32



    # Using the updated create_dataset function to generate sequences
    train_X_full, train_Y_full = create_dataset(df_dynamic[(df_dynamic['date_id'].between(0,477))], TIME_STEPS)
    test_X, test_Y = create_dataset(df_dynamic[(df_dynamic['date_id'].between(478, 480))],TIME_STEPS)

    # Split the sequences of the training data into training and validation
    train_X, val_X, train_Y, val_Y = train_test_split(train_X_full, train_Y_full, test_size=0.2, random_state=42)

    print(train_X.shape, train_Y.shape)
    print(val_X.shape, val_Y.shape)
    print(test_X.shape, test_Y.shape)

    # Assuming attention_model is defined elsewhere in your code
    m = attention_model()
    m.summary()

    optimizer = Adam(learning_rate=0.001) 
    BATCH_SIZE = 64 * tpu_strategy.num_replicas_in_sync 
    m.compile(loss='mse', optimizer=optimizer, metrics=['mae'], steps_per_execution=32)  # Added steps_per_execution for TPU optimization

    # Fit the LSTM model
    history = m.fit([train_X], train_Y, epochs=100, batch_size=BATCH_SIZE, validation_data=([val_X], val_Y))

    # After training, you can evaluate your model on the test dataset
    test_loss, test_mae = m.evaluate([test_X], test_Y, batch_size=BATCH_SIZE)
    print(f"Test MAE: {test_mae}")