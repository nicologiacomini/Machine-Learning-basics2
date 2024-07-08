import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import LSTM, Dense 
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import os


def polynomial_interpolation(data):
    indices = np.arange(len(data))
    not_nan = ~np.isnan(data)
    
    interp_func = interp1d(indices[not_nan], data[not_nan], kind='cubic', fill_value="extrapolate")
    imputed_data = interp_func(indices)
    
    return imputed_data

def create_lstm_model(window_size):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(window_size, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def prepare_lstm_data(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    X, y = np.array(X), np.array(y)
    return X, y

def lstm_imputation(data, window_size):
    data_filled = pd.Series(data).fillna(method='ffill').fillna(method='bfill').values

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data.reshape(-1, 1))
    
    X, y = prepare_lstm_data(data_scaled, window_size)
    
    model = create_lstm_model(window_size)
    model.fit(X, y, epochs=200, verbose=0)
    
    imputed_data_scaled = model.predict(X)
    imputed_data = scaler.inverse_transform(imputed_data_scaled).flatten()
    
    # Reconstruct the complete data sequence
    imputed_complete = np.copy(data)
    imputed_complete[window_size:] = imputed_data
    return imputed_complete

def evaluate_imputation(true_values, imputed_values):
    rmse = np.sqrt(mean_squared_error(true_values, imputed_values))
    r2 = r2_score(true_values, imputed_values)
    return rmse, r2

def main(filename):
    # Load the datasets
    missing_data_df = pd.read_csv(f'data/{filename}', sep=';')
    complete_data_df = pd.read_csv('data_matrix.csv', sep=';')
    sensors = complete_data_df.columns.drop('date')
    
    results = []

    for sensor in sensors:
        perc = float(filename.split('-')[1])
        burst = int(filename.split('-')[2].split('.csv')[0])
        complete_values = complete_data_df[sensor].values
        missing_values = missing_data_df[sensor].values

        # Polynomial interpolation
        poly_imputed = polynomial_interpolation(missing_values)
        poly_rmse, poly_r2 = evaluate_imputation(complete_values, poly_imputed)
        
        # Substitute all the NaN values with 0
        missing_values[np.isnan(missing_values)] = 0

        # LSTM imputation
        lstm_imputed = lstm_imputation(missing_values, window_size=100)  # example window size
        lstm_rmse, lstm_r2 = evaluate_imputation(complete_values, lstm_imputed)
        
        results.append({
            'sensor': sensor,
            'missing_percentage': perc,
            'burst_length': burst,
            'poly_rmse': poly_rmse,
            'poly_r2': poly_r2,
            'lstm_rmse': lstm_rmse,
            'lstm_r2': lstm_r2
        })
    
    results_df = pd.DataFrame(results)
    filename += '-T100.csv'
    results_df.to_csv(f'results/unvariate/{filename}', index=False)
    print(results_df)

if __name__ == '__main__':
    for _, _, files in os.walk('data'):
        for file in files:
            main(file)
