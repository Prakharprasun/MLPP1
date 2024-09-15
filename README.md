# Stock-Predictor-vMLPP1
We are trying to predict the closing price of stock using  ML, particularly historical data and an LSTM (Long Short-Term Memory) neural network. The model is built in Python using Quandl, Numpy, Pandas, Matplotlib, Tensorflow and Sklearn for data acquisition ,data visualisation and deep learning.

# Overview
Predicting stock prices is a complex task due to the volatile and non-stationary nature of financial time series data. This project leverages LSTM neural networks, which are particularly well-suited for time series prediction due to their ability to capture long-term dependencies and temporal structures in the data.
Recurrent neural networks (RNN) are a class of neural networks that is powerful for modelling sequence data such as time series or natural language.

Schematically, a RNN layer uses a for loop to iterate over the timesteps of a sequence, while maintaining an internal state that encodes information about the timesteps it has seen so far.
Stock data by nature is unpredictable, this is merely an attempt to learn Machine Learning by trying to predict it.
# Data Acquisition
We use Quandl to fetch stock data for Tata Global Beverages (TATAGLOBAL) from the National Stock Exchange (NSE). The dataset contains historical prices such as Open, High, Low, Close, Volume, and Adjusted values.
# Data Analysis
Before building the model, we visualise the stock's features mainly via plot to identify trends, volatility, and patterns. This includes features like Open, Close, High, Low, and Volume.
# Technical Indicators
We compute two essential Simple Moving Averages (SMA):
50-day SMA (short-term trend)
200-day SMA (long-term trend)
These technical indicators are used to smooth the price data and identify buy/sell signals.
# Preprocessing
To ensure the data is in a suitable format for our LSTM model:

Scaling: We normalise the data using StandardScaler to ensure all features have a mean of 0 and a standard deviation of 1.

Splitting: The dataset is split into training, validation, and test sets.

Scaling your data in machine learning (ML)is important because many algorithms use the Euclidean distance between two data points in their computations/derivations, which is sensitive to the scale of the variables. If one variable is on a much larger scale than another, that variable will dominate the distance calculation, and the algorithm will be affected by that variable more than the other. Scaling the data can help to balance the impact of all variables on the distance calculation and can help to improve the performance of the algorithm. In particular, several ML techniques, such as neural networks, require that the input data be normalised for it to work well.

There are several libraries in Python that can be used to scale data:
Standardisation: The mean of each feature becomes 0 and the standard deviation becomes 1.
Normalisation: The values of each feature are between 0 and 1.
Min-Max Scaling: The minimum value of each feature becomes 0 and the maximum value becomes 1.
# Model
The LSTM neural network is chosen for its ability to capture long-term dependencies in time series data. The architecture consists of:

Input Layer

LSTM layer with 64 units

Dense layer with 32 units

Dense layer with 32 units

Dense layer with 1

The model uses Mean Squared Error (MSE) as the loss function and the Adam optimizer for training.
Optimizers are algorithms or methods that are used to change or tune the attributes of a neural network such as layer weights, learning rate, etc. in order to reduce the loss and in turn improve the model.

Adam(Adaptive Moment Estimation) is an adaptive optimization algorithm that was created specifically for deep neural network training. It can be viewed as a fusion of momentum-based stochastic gradient descent and RMSprop. It scales the learning rate using squared gradients, similar to RMSprop, and leverages momentum by using the gradient’s moving average rather than the gradient itself, similar to SGD with momentum.
Training and Validation
The model is trained using the training set and evaluated on the validation set for 50 epochs using Mean Absolute Error (MAE) as a metric.
# Evaluation
After training, the model's performance is evaluated on both the training and test sets, on:
Train Loss
Validation Loss
Test Loss
Mean Absolute Error (MAE) for each set
# References
Tensorflow

GeeksforGeeks

Medium


