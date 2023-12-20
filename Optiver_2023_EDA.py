# %% [markdown]
# # Optiver Trading at the Close: Preprocessing and Feature Engineering for Time Series Forecasting
# 
# 
# Welcome to this notebook where we tackle a complex yet highly fascinating task: Forecasting the 60-second weighted average price (WAP) change against the 60-second synthetic stock index change. This exercise is particularly crucial in understanding market dynamics and making informed trading decisions.
# 
# ## The Problem
# 
# The problem at hand is a time series forecasting challenge that offers a robust dataset featuring over 200 stock IDs across 500 trading days. The goal is to predict short-term market movements to facilitate better trading decisions. Notably, our model will be evaluated on a private leaderboard with unseen data after the competition ends.
# 
# ## The Data
# 
# The data for this task is rich and complex, covering various aspects of stock trading. It offers insights into order imbalances, bid and ask prices, weighted average prices, and much more. 
# 
# ## Order Imbalance
# 
# One of the key elements we examine in this notebook is "Order Imbalance." This occurs when there is a significant skew between buy and sell orders for a given stock during the 10 minutes leading up to the closing auction in the U.S stock market. Understanding the imbalance helps in predicting the directional movement of a stock’s price.
# 
# ## Exploratory Data Analysis (EDA)
# 
# We begin our journey with a comprehensive Exploratory Data Analysis (EDA) to get a deeper understanding of the data's nuances. This will inform the feature engineering steps and ultimately guide our choice of forecasting model.
# 
# ## Objective
# 
# By the end of this notebook, you will have a better grasp of the data's structure, the significance of order imbalance, and the various technical indicators that can aid in forecasting stock price movements.
# 
# So, let's dive in and unravel the intricacies of financial time series forecasting!
# 

# %% [code] {"execution":{"iopub.status.busy":"2023-10-30T17:29:59.252442Z","iopub.execute_input":"2023-10-30T17:29:59.252867Z","iopub.status.idle":"2023-10-30T17:30:02.656512Z","shell.execute_reply.started":"2023-10-30T17:29:59.252819Z","shell.execute_reply":"2023-10-30T17:30:02.653835Z"},"jupyter":{"outputs_hidden":true}}
######################
# Libraries and Data #
######################

# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression


# Read in data
df= pd.read_csv('/kaggle/input/optiver-trading-at-the-close/train.csv')


# %% [code] {"execution":{"iopub.status.busy":"2023-10-30T17:30:02.657548Z","iopub.status.idle":"2023-10-30T17:30:02.657960Z","shell.execute_reply.started":"2023-10-30T17:30:02.657758Z","shell.execute_reply":"2023-10-30T17:30:02.657777Z"}}

############
# Metadata #
############

# Print Data
print(df.head(5))

# Get Meta
df.info()



# %% [markdown]
# 
# 
# ## General Observations
# - The dataset contains 5,237,980 entries and 17 columns.
# - The columns include a mix of integer, float, and object types.
# 
# ## Features Description
# 
# ### Time-Related Features
# 1. **Stock ID**: Identifies the stock. In the given snippet, it ranges from 0 to 4, but likely covers a larger range given the dataset's size.
# 2. **Date ID and Time ID**: Crucial for time-series analysis.
# 3. **Seconds in Bucket**: Represents time elapsed within a given trading day and is another essential feature for time-series analysis.
# 
# ### Order Book Metrics
# 4. **Imbalance Size**: Represents the size of imbalance in orders. The values are significantly different, suggesting different trading activities for different stocks.
# 5. **Imbalance Buy/Sell Flag**: A flag indicating whether the imbalance comes from buying (1) or selling (-1).
# 6. **Matched Size**: Size that has been matched in the order book.
# 
# ### Price Points
# 7. **Reference Price, Far Price, Near Price, Bid Price, Ask Price**: Various prices related to the order book. Notably, 'Far Price' and 'Near Price' have missing (NaN) values that may require imputation.
# 
# ### Market Metrics
# 8. **Bid Size, Ask Size**: Represents the size of bids and asks in the order book, respectively.
# 9. **WAP (Weighted Average Price)**: A feature that takes into account both price and volume.
# 
# ### Target Variable
# 10. **Target**: Appears to be the target variable for a predictive model, being continuous in nature.
# 
# ### Data Types
# - Integer types: `stock_id`, `date_id`, `seconds_in_bucket`, `imbalance_buy_sell_flag`, `time_id`
# - Float types: `imbalance_size`, `reference_price`, `matched_size`, `far_price`, `near_price`, `bid_price`, `ask_size`, `wap`, `target`
# - Object types: `row_id`
# 
# ## Possible Points of Action
# - **Missing Value Imputation**: Consider imputing missing values for 'Far Price' and 'Near Price' based on the time-series nature of the data.
# - **Feature Scaling**: Normalize or standardize numerical features especially if using algorithms sensitive to feature scaling.
# - **Lag Features**: Consider creating lag features based on `time_id` and `seconds_in_bucket` for time-series analysis.
# 

# %% [code] {"execution":{"iopub.status.busy":"2023-10-30T17:30:02.660144Z","iopub.status.idle":"2023-10-30T17:30:02.660639Z","shell.execute_reply.started":"2023-10-30T17:30:02.660415Z","shell.execute_reply":"2023-10-30T17:30:02.660437Z"}}

#########################
# Descriptive Statstics #
#########################


# Data Columns and Types
# Show basic statistics for univariate analysis
print("Basic Statistics:")
print(df.describe())

# Check for missing values
print("Missing Values:")
print(df.isnull().sum())






# %% [code] {"execution":{"iopub.status.busy":"2023-10-30T17:30:02.661960Z","iopub.status.idle":"2023-10-30T17:30:02.662413Z","shell.execute_reply.started":"2023-10-30T17:30:02.662181Z","shell.execute_reply":"2023-10-30T17:30:02.662201Z"}}
df.groupby('stock_id').count()


# %% [markdown]
# We can see that we expect that the training set has recorded targets for dates from 0-480 and within each day there is a recorded value at each 10 seconds time step for each bucket.

# %% [code] {"execution":{"iopub.status.busy":"2023-10-30T17:30:02.664059Z","iopub.status.idle":"2023-10-30T17:30:02.664508Z","shell.execute_reply.started":"2023-10-30T17:30:02.664274Z","shell.execute_reply":"2023-10-30T17:30:02.664293Z"}}

agg_df=df.groupby('stock_id').agg(
    date_id_min=('date_id', 'min'), 
    date_id_max=('date_id', 'max'), 
    seconds_min=('seconds_in_bucket', 'min'), 
    seconds_max=('seconds_in_bucket', 'max')
)


# Assume that the 'standard_row' is the row you consider to be the standard for comparison.
standard_row = agg_df.iloc[0]

# Find rows that are different from the standard.
different_rows = agg_df[
    (agg_df['date_id_min'] != standard_row['date_id_min']) | 
    (agg_df['date_id_max'] != standard_row['date_id_max']) | 
    (agg_df['seconds_min'] != standard_row['seconds_min']) | 
    (agg_df['seconds_max'] != standard_row['seconds_max'])
]

print(different_rows)



# %% [markdown]
# Some ambiguities is that some stocks have different starting dates, in which closing target is recorded  

# %% [code] {"execution":{"iopub.status.busy":"2023-10-30T17:30:02.666164Z","iopub.status.idle":"2023-10-30T17:30:02.666599Z","shell.execute_reply.started":"2023-10-30T17:30:02.666399Z","shell.execute_reply":"2023-10-30T17:30:02.666420Z"}}
unique_counts = df.groupby('stock_id')[['date_id', 'seconds_in_bucket']].nunique().reset_index()

# Assume 'standard_row' is the row you consider to be the standard for comparison.
standard_row = unique_counts.iloc[0]

# Find rows that differ from the standard
different_rows = unique_counts[(unique_counts['date_id'] != standard_row['date_id']) | (unique_counts['seconds_in_bucket'] != standard_row['seconds_in_bucket'])]

print(different_rows)



# %% [markdown]
# We can see here that some stocks specifically those lsited above , have incomplete dates .

# %% [code] {"execution":{"iopub.status.busy":"2023-10-30T17:30:02.668064Z","iopub.status.idle":"2023-10-30T17:30:02.668518Z","shell.execute_reply.started":"2023-10-30T17:30:02.668280Z","shell.execute_reply":"2023-10-30T17:30:02.668301Z"}}
#Check Uniformity 
expected_seconds = list(range(0, 541, 10))

# Group by 'stock_id' and get unique 'seconds_in_bucket' values for each group
grouped = df.groupby('stock_id')['seconds_in_bucket'].unique().reset_index()

# Initialize a list to keep track of stock_ids that do not have complete data
stocks_incomplete = []

# Loop through each stock_id and check if the unique 'seconds_in_bucket' values match the expected list
for idx, row in grouped.iterrows():
    stock_id = row['stock_id']
    unique_seconds = sorted(row['seconds_in_bucket'])
    if unique_seconds != expected_seconds:
        stocks_incomplete.append(stock_id)
        
print("Stock IDs with incomplete seconds in uckets recorded values:", stocks_incomplete)


# %% [code] {"execution":{"iopub.status.busy":"2023-10-30T17:30:02.671192Z","iopub.status.idle":"2023-10-30T17:30:02.672536Z","shell.execute_reply.started":"2023-10-30T17:30:02.672277Z","shell.execute_reply":"2023-10-30T17:30:02.672298Z"}}
#Check Uniformity
expected_dates = list(range(0, 481))

# Group by 'stock_id' and get unique 'date_id' values for each group
grouped_dates = df.groupby('stock_id')['date_id'].unique().reset_index()

# Initialize a list to keep track of stock_ids that do not have complete data
stocks_incomplete_dates = []

# Loop through each stock_id and check if the unique 'date_id' values match the expected list
for idx, row in grouped_dates.iterrows():
    stock_id = row['stock_id']
    unique_dates = sorted(row['date_id'])
    if unique_dates != expected_dates:
        stocks_incomplete_dates.append(stock_id)

print("Stock IDs with incomplete data for date IDs:", stocks_incomplete_dates)



# %% [markdown]
# # Summary Statistics and Missing Value Analysis 
# 
# ## Summary Statistics:
# 
# ### Overall Descriptions
# - The dataset contains over 5 million entries.
# - The features show significant variability both in terms of mean and standard deviation.
# - Some features like `imbalance_size` and `matched_size` show a large range and high standard deviation, suggesting the potential for outliers or extreme values.
# 
# ### Noteworthy Points
# 1. `imbalance_size` has a wide range, with a max value of approximately 2.98 billion and a minimum of 0.
# 2. `far_price` and `near_price` show a standard deviation considerably lower than their mean, suggesting less volatility.
# 3. The target variable (`target`) has a mean close to 0 but a standard deviation of approximately 9.45, indicating a wide range of values.
# 
# ## Missing Values:
# 
# ### Count of Missing Values by Column
# - `imbalance_size`, `reference_price`, `matched_size`, `bid_price`, `ask_price`, `wap`: 220 missing values
# - `far_price`: 2,894,342 missing values
# - `near_price`: 2,857,180 missing values
# - `target`: 88 missing values
# 
# ### Proposed Solutions for Missing Values:
# 
# 1. **For `imbalance_size`, `reference_price`, `matched_size`, `bid_price`, `ask_price`, `wap`**:
#     - Since these features are time-series in nature and the number of missing values is small, we can use time-based imputation methods like forward-fill or back-fill based on `time_id` and `date_id`, or linear implemntation to preserve the context of timerseries.
# 
# 2. **For `far_price` and `near_price`**:
#     - These columns have a high number of missing values. One possible approach could be to fill these using interpolation methods tailored for time-series data, given the `time_id` and `date_id`.
#     - Alternatively, you can use more complex imputation techniques like k-Nearest Neighbors imputation or even model-based imputation where the missing values are predicted based on other observed variables.
# 
# 3. **For `target`**:
#     - If `target` is the variable you intend to predict, then it's best to remove these rows from the dataset as they won't be useful for training or testing the model.
# 
# ## Next Steps:
# 
# - Perform the imputations as per the proposed solutions.
# - After imputation, re-check the summary statistics to ensure that the imputations haven't drastically changed the dataset's characteristics.
# - Given that the data is high-dimensional and time-series in nature, consider using dimensionality reduction techniques and feature engineering to improve the model's performance for forecasting the stock price at closing day.
# 
# 

# %% [code] {"execution":{"iopub.status.busy":"2023-10-30T17:30:02.673568Z","iopub.status.idle":"2023-10-30T17:30:02.674321Z","shell.execute_reply.started":"2023-10-30T17:30:02.674096Z","shell.execute_reply":"2023-10-30T17:30:02.674125Z"}}

#######################
# Univaraite Analysis #
#######################


# Univariate Analysis: Histograms for numerical columns
for col in df.select_dtypes(include=[np.number]).columns:
    plt.figure()
    sns.histplot(df[col], bins=20, kde=False)
    plt.title(f'Histogram of {col}')
    plt.show()


# %% [code] {"execution":{"iopub.status.busy":"2023-10-30T17:30:02.675888Z","iopub.status.idle":"2023-10-30T17:30:02.676444Z","shell.execute_reply.started":"2023-10-30T17:30:02.676145Z","shell.execute_reply":"2023-10-30T17:30:02.676170Z"}}
#######################
# Bivariate Analysis #
#######################

# Scatter plots for each numerical feature against 'target'
for col in df.select_dtypes(include=[np.number]).columns:
    if col != 'target':  # Skip the 'target' column itself
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=col, y='target', data=df)
        plt.title(f'Scatter Plot of {col} vs Target')
        plt.xlabel(col)
        plt.ylabel('Target')
        plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2023-10-30T17:30:02.678225Z","iopub.status.idle":"2023-10-30T17:30:02.678967Z","shell.execute_reply.started":"2023-10-30T17:30:02.678753Z","shell.execute_reply":"2023-10-30T17:30:02.678773Z"}}
# Calculate CV for each column
cv = {}
for col in df.select_dtypes(include=['float64', 'int64']).columns:
    mean_value = df[col].mean()
    std_value = df[col].std()
    
    if mean_value != 0:  # To avoid division by zero
        cv[col] = (std_value / mean_value) * 100

print(cv)

# %% [markdown]
# # Coefficient of Variation Analysis
# 
# ## Interpretation of Coefficient of Variation (CV)
# 
# The CV for each column gives us an idea of the variability of the data in relation to the mean of the column. 
# 
# ### High CV (>100%)
# 
# - **imbalance_size**: 358.97%
# - **matched_size**: 310.07%
# - **imbalance_buy_sell_flag**: -7442.19%
# - **target**: -19875.13%
# 
# These columns have high variability compared to their mean. Transformations such as log, square root, or even binning might be useful.
# 
# ### Moderate CV (10% - 100%)
# 
# - **stock_id**: 58.29%
# - **date_id**: 57.36%
# - **seconds_in_bucket**: 58.79%
# - **time_id**: 57.24%
# - **far_price**: 72.02%
# - **bid_size**: 215.04%
# - **ask_size**: 241.44%
# 
# These columns have moderate variability. You might still consider some transformations depending on the modeling algorithm you plan to use and the specific relationship between the feature and the target variable.
# 
# ### Low CV (<10%)
# 
# - **reference_price**: 0.25%
# - **near_price**: 1.22%
# - **bid_price**: 0.25%
# - **ask_price**: 0.25%
# - **wap**: 0.25%
# 
# These columns have low variability compared to their mean. They may not need a transformation for stabilizing variance, but scaling might still be necessary depending on the modeling technique you're using.
# 
# ## Recommendations for Transformations
# 
# 1. **High CV columns**: For features like `imbalance_size`, `matched_size`, `imbalance_buy_sell_flag` consider logarithmic or square root transformations to stabilize the high variance. Note that log transformations are only applicable to positive numbers.
# 
# 2. **Moderate CV columns**: Features like `stock_id`, `date_id`, `seconds_in_bucket`, and `time_id` may or may not need transformations. This could be better determined by looking at their relationship with the target variable, possibly through scatter plots or correlation matrices. We know that `date_id`, `seconds_in_bucket`, and `time_id` form as time indcators so we dont need to transform them.
# 
# 3. **Low CV columns**: These usually don't need transformations for variance stabilization, but you might still want to scale them if your model is sensitive to the magnitude of input variables (like SVM, KNN, etc.).
# 
# 
# 

# %% [code] {"execution":{"iopub.status.busy":"2023-10-30T17:30:02.680083Z","iopub.status.idle":"2023-10-30T17:30:02.681259Z","shell.execute_reply.started":"2023-10-30T17:30:02.680967Z","shell.execute_reply":"2023-10-30T17:30:02.680994Z"}}

# Multivariate Analysis: Correlation matrix
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=False)
plt.title("Correlation Matrix")
plt.show()

   
print(corr_matrix)

# %% [markdown]
# # Observations and Considerations on Correlation Matrix
# 
# ## Observations:
# 
# ### 1. Highly Correlated Prices
# The columns `reference_price`, `bid_price`, `ask_price`, and `wap` (weighted average price) are highly correlated with each other. Their correlation coefficients are close to 1. This suggests that they contain very similar information.
# 
# ### 2. `matched_size` and `imbalance_size`
# These two variables have a correlation of 0.51, which is moderate. This means they share some information but not entirely.
# 
# ### 3. Time-related Correlations
# `bid_size`, `ask_size`, and `seconds_in_bucket` show moderate correlation. This suggests that these variables are related to the time buckets but not so strongly.
# 
# ### 4. `near_price` and `imbalance_buy_sell_flag`
# With a correlation of approximately 0.51, this pair seems to be somewhat related.
# 
# ### 5. Date and Time IDs
# Both `date_id` and `time_id` are highly correlated (0.999998), which makes sense since both are time identifiers. However, they don't seem to be strongly correlated with any other feature.
# 
# ### 6. Low Correlation with Target
# None of the variables seem to have a strong correlation with `target`, which might indicate that predicting the target variable directly using these features might be challenging.
# 
# ## Considerations and Solutions:
# 
# ### 1. Feature Reduction
# Since `reference_price`, `bid_price`, `ask_price`, and `wap` are highly correlated, you may consider keeping just one or create a feature that captures their commonality to reduce multicollinearity.
# 
# ### 2. Feature Engineering
# For moderately correlated variables, you can think about creating interaction terms or ratios to better capture the relationships.
# 
# ### 3. Time-Related Features
# If `seconds_in_bucket`, `bid_size`, and `ask_size` are important, consider creating aggregated features based on time buckets.
# 
# ### 4. Target Prediction
# Given that the correlation with the target is low, consider ensemble methods, feature engineering, or more complex models to capture the underlying patterns.
# 
# ### 5. Ignore IDs
# As mentioned, IDs like `stock_id`, `date_id`, `time_id`, and `row_id` are identifiers and should generally not be used in the modeling process unless they capture some form of seasonality or trend.
# 
# ### 6. Additional Data
# Since none of the features are strongly correlated with the target, obtaining additional data or features could be beneficial for the prediction task.
# 
# ### 7. Normalization
# Before proceeding with models like k-NN or neural networks, consider normalizing the features, as they appear to be on different scales based on the magnitude of the correlation coefficients.
# 
# > **Note:** Remember, correlation does not imply causation. Therefore, it's essential to understand the business context of the data for better feature selection and engineering.
# 

# %% [code] {"execution":{"iopub.status.busy":"2023-10-30T17:30:02.683211Z","iopub.status.idle":"2023-10-30T17:30:02.683783Z","shell.execute_reply.started":"2023-10-30T17:30:02.683499Z","shell.execute_reply":"2023-10-30T17:30:02.683534Z"}}
#######################
# Additional Analysis #
#######################

# Plotting Buy/Sell/No Pressure states
plt.figure(figsize=(15, 6))

# Assuming 'imbalance_buy_sell_flag' is the column representing buy/sell/no pressure
# +1 for buy, -1 for sell, 0 for no pressure
sns.lineplot(data=df, x='date_id', y='imbalance_buy_sell_flag', estimator='mean')

plt.axvline(40, color='r', linestyle='--', label='Day 40')
plt.axvline(220, color='g', linestyle='--', label='Day 220')
plt.axvline(355, color='b', linestyle='--', label='Day 355')

plt.legend()
plt.title('Buy/Sell/No Pressure over Days')
plt.show()

# Check if not all stocks are traded on all days
unique_dates = df['date_id'].unique()
for stock_id in df['stock_id'].unique():
    stock_dates = df[df['stock_id'] == stock_id]['date_id'].unique()
    missing_dates = set(unique_dates) - set(stock_dates)
    if missing_dates:
        print(f"Stock ID {stock_id} is missing on Date IDs: {missing_dates}")

# For identifying outliers in trading volumes (assuming 'matched_size' represents trading volume)
sns.boxplot(x=df['matched_size'])
plt.title('Outliers in Trading Volume')
plt.show()
