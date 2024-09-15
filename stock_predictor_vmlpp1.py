#use!pip install quandl if not already installed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
import quandl

class StockPredictor:
    def __init__(self, stock_code, scaler=StandardScaler()):
        #Initializes the StockPredictor with stock data from Quandl and a scaler.      
        #Args:
        #stock_code (str): The Quandl stock code for the desired stock data.
        #scaler (object): The scaler object to normalize the data (default: StandardScaler).
        self.stock_code = stock_code
        self.data = self.get_stock_data()
        self.scaler = scaler
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = None, None, None, None, None, None
        self.model = None

    def get_stock_data(self):
        #Fetches stock data from Quandl using the provided stock code.       
        #Returns: pd.DataFrame: Stock data with various features.
        data = quandl.get(self.stock_code)
        print(data.head(10))
        return data

    def plot_features(self):
        #Plots all features from the stock data to visualize trends over time.
        for feature in self.data.columns:
            plt.figure(figsize=(16, 9))
            plt.xlabel("Date")
            plt.ylabel(feature)
            plt.plot(self.data[feature])
            plt.title(f"{feature} over Time")
            plt.show()

    def calculate_sma(self):
        #Calculates Simple Moving Averages (SMA) for 50 and 200 days and plots them.
        self.data['sma_50'] = self.data['Close'].rolling(window=50).mean()
        self.data['sma_200'] = self.data['Close'].rolling(window=200).mean()

        plt.figure(figsize=(16, 9))
        plt.plot(self.data['Close'], label='Close Price')
        plt.plot(self.data['sma_50'], label='50-day SMA')
        plt.plot(self.data['sma_200'], label='200-day SMA')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title('SMA 50 and 200 Days')
        plt.legend()
        plt.show()

    def prepare_data(self):
        #Prepares data by scaling features and splitting into training, validation, and test sets.
        #Scaling the features
        df = pd.DataFrame(self.data.values, columns=self.data.columns)
        df_scaled = pd.DataFrame(self.scaler.fit_transform(df), columns=df.columns)

        #Target and features
        X = df_scaled.drop('Close', axis=1)
        y = df['Close']

        #Splitting the data
        self.X_train, self.y_train = X[1500:], y[1500:]
        self.X_val, self.y_val = X[1500:1700], y[1500:1700]
        self.X_test, self.y_test = X[1700:], y[1700:]

        #Reshaping for LSTM input
        self.X_train = np.expand_dims(self.X_train, axis=-1)
        self.X_val = np.expand_dims(self.X_val, axis=-1)
        self.X_test = np.expand_dims(self.X_test, axis=-1)

    def build_model(self):
        #Builds an LSTM model for stock price prediction.
        self.model = Sequential([
            layers.Input(shape=(self.X_train.shape[1], self.X_train.shape[2])),
            layers.LSTM(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)
        ])
        self.model.compile(loss='mse', optimizer=Adam(learning_rate=0.001), metrics=['mean_absolute_error'])

    def train_model(self, epochs=50):
        #Trains the LSTM model with the prepared data.
        #Args: epochs (int): Number of training epochs (default: 50).

        self.model.fit(self.X_train, self.y_train, validation_data=(self.X_val, self.y_val), epochs=epochs)

    def evaluate_model(self, dataset='train'):

        #Evaluates the model's performance on the specified dataset (train/val/test).        
        #Args: dataset (str): The dataset to evaluate ('train', 'val', 'test').            
        #Returns: tuple: Loss and MAE for the specified dataset.
        if dataset == 'train':
            return self.model.evaluate(self.X_train, self.y_train)
        elif dataset == 'val':
            return self.model.evaluate(self.X_val, self.y_val)
        elif dataset == 'test':
            return self.model.evaluate(self.X_test, self.y_test)

    def plot_predictions(self, dataset='train'):
        #Plots the model's predictions against the actual data for a given dataset.        
        #Args: dataset (str): The dataset to plot ('train', 'val', 'test').

        if dataset == 'train':
            predict = self.model.predict(self.X_train)
            actual = self.y_train.values
        elif dataset == 'val':
            predict = self.model.predict(self.X_val)
            actual = self.y_val.values
        elif dataset == 'test':
            predict = self.model.predict(self.X_test)
            actual = self.y_test.values

        plt.figure(figsize=(16, 9))
        plt.plot(predict, label="Predicted")
        plt.plot(actual, label="Actual")
        plt.title(f"{dataset.capitalize()} Set Predictions vs Actual")
        plt.legend()
        plt.show()


# Instantiate the StockPredictor class for a specific stock and perform analysis
if __name__ == "__main__":
    predictor = StockPredictor("NSE/TATAGLOBAL")
    predictor.plot_features()
    predictor.calculate_sma()
    predictor.prepare_data()
    predictor.build_model()
    predictor.train_model(epochs=50)

    # Plot and evaluate predictions
    predictor.plot_predictions(dataset='train')
    predictor.plot_predictions(dataset='val')
    predictor.plot_predictions(dataset='test')

    print("Train Performance:", predictor.evaluate_model(dataset='train'))
    print("Validation Performance:", predictor.evaluate_model(dataset='val'))
    print("Test Performance:", predictor.evaluate_model(dataset='test'))
