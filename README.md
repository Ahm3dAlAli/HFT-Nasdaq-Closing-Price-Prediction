# 📈 Optiver Trading at the Close: Time Series Forecasting for Financial Markets

Welcome to the Optiver Trading at the Close. This project focuses on forecasting the 60-second weighted average price (WAP) change against the 60-second synthetic stock index change, utilizing advanced machine learning and deep learning techniques.

## 📋 Table of Contents

- [Overview](#overview)
- [Data Analysis](#data-analysis)
- [Feature Engineering](#feature-engineering)
- [Modeling and Training](#modeling-and-training)
- [Model Evaluation](#model-evaluation)
- [Deep Learning Model](#deep-learning-model)
- [Data Preprocessing](#data-preprocessing)
- [API Testing](#api-testing-and-modeling)

<a name="overview"></a>
## 🌟 Overview

The project involves intricate data analysis, preprocessing, feature engineering, and the deployment of sophisticated models like XGBoost and CNN-BiLSTM-ATN for time series prediction in the financial domain.

<a name="data-analysis"></a>
## 📊 Data Analysis

The initial phase focuses on exploratory data analysis (EDA) to understand the intricacies of the dataset, which comprises over 5 million entries and 17 columns, including time-related data, order book metrics, and market metrics.

<a name="feature-engineering"></a>
## 🧪 Feature Engineering

In-depth feature engineering was conducted to enhance the model's predictive power. This includes:
- **Imbalance Features**: Calculating imbalance direction, book imbalance, and price imbalance.
- **Order Book Features**: Including bid-ask spread, mid-price movement, normalized spread, and size ratio.
- **Order Flow and Volume Features**: Tracking volume changes and daily volatility.
- **Pricing Features**: Focusing on reference to WAP ratio, overnight returns, and more.
- **Technical Indicators**: Applying MACD, RSI, RoC13, Bollinger Bands, and others.
- **Transformation Techniques**: Utilizing linear interpolation, linear regression, and various transformations for missing value imputation and feature optimization.

<a name="modeling"></a>
## 🤖 Modeling and Training

### XGBoost Model
- Parameters: Learning rate, max depth, min child weight, colsample by tree, subsample.
- Bayesian Hyperparameter Tuning for enhanced performance.
- Model Evaluation using Mean Absolute Error (MAE).

### Deep Learning Model (CNN-BiLSTM-ATN)
- **Architecture**: Combines convolutional layers, bidirectional LSTM layers, and attention mechanisms.
- **Training**: Utilizes Adam optimizer with a learning rate of 0.001 and TPU execution for efficiency.
- **Layers**: Includes Conv1D, BatchNormalization, Dropout, Bidirectional LSTM, and Dense layers.
- **Hyperparameters**: Batch size dynamically adjusted for TPU execution, dropout rates for regularization.
- **Specifics**: The model was trained with 82 input dimensions and 54 time steps. LSTM units were set to 32. The model was compiled with a mean squared error loss function and evaluated using MAE.

<a name="evaluation"></a>
## 🎯 Model Evaluation

- The XGBoost model showed significant performance, optimized through Bayesian Hyperparameter Tuning, achieving a Mean Absolute Error (MAE) of 2.46999 on the test set and about 3.0 during training, indicating its effectiveness in time series forecasting.

<a name="deep-learning-model"></a>
## 🧠 Deep Learning Model (CNN-BiLSTM-ATN)

- **Architecture**: Integrates convolutional layers for feature extraction, bidirectional LSTM layers for capturing temporal dependencies, and attention mechanisms for focusing on relevant time steps.
- **Training Environment**: Utilizes Tensor Processing Units (TPUs) for efficient and faster training processes.
- **Model Serialization**: The trained models are saved for future use, ensuring reproducibility and deployment readiness.

<a name="data-preprocessing"></a>
## 🔄 Data Preprocessing

- Data scaling and transformation using MinMaxScaler.
- Outlier detection and handling.
- Lag features and rolling window statistics for capturing temporal trends.
- Data partitioning into training, validation, and test sets.

<a name="api-testing-and-modeling"></a>
## 🌐 API Testing and Modeling

During the API testing phase, a mock loop was established to handle incoming test data. The test data was processed and prepared for prediction using either the Neural or XGBoost models, depending on the `Modelling` flag.

- **Neural Model Processing**: Test data is scaled using a saved MinMaxScaler model. Predictions are generated using the CNN-BiLSTM-ATN model, with reshaped data to match the model's input shape.
- **XGBoost Model Processing**: Features are prepared, and predictions are made using the XGBoost model.
- **Dynamic Data Update**: The training dataset is dynamically updated with new predictions and sorted for consistency.

This approach ensures that the model stays up-to-date with the latest market trends and adapts its predictions accordingly.
