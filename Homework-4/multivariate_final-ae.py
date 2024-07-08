import pandas as pd
import os
from math import radians, cos, sin, asin, sqrt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

def build_autoencoder(input_dim, encoding_dim):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation='relu')(input_layer)
    decoded = Dense(input_dim, activation='sigmoid')(encoded)
    
    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    
    return autoencoder

def main(filename, index):
    # Step 0: load the datasets
    missing_data_df = pd.read_csv(f'data/{filename}', sep=';').drop(columns=['date'])
    complete_data_df = pd.read_csv('data_matrix.csv', sep=';').drop(columns=['date'])

    # imputed_data = mice_imputation(missing_data_df)

    # Step 2: Calculate the RMSE and R2
    sensors_list = ["gracia", "eixample", "pr", "hebron", "ciutadella", "badalona", "montcada", "prat"]
    results = []
    df = missing_data_df[sensors_list[0]]
    complete_values = complete_data_df[sensors_list[0]]
    imputed_data = complete_data_df[sensors_list[0]]
    sensors_final = [sensors_list[0]]

    for sensor in sensors_list[1:]:
        # add to df missing_data_df[sensor]
        df = pd.concat([df, missing_data_df[sensor]], axis=1)
        mean_imputed_data = df.fillna(df.mean())

        complete_values = pd.concat([complete_values, complete_data_df[sensor]], axis=1)
        
        input_dim = mean_imputed_data.shape[1]
        encoding_dim = df.shape[1]  # For example, using 3 as the dimension of latent space

        autoencoder = build_autoencoder(input_dim, encoding_dim)
        autoencoder.fit(mean_imputed_data, mean_imputed_data, epochs=1000, batch_size=2257, shuffle=False)

        # Imputation with AE
        reconstructed_data = autoencoder.predict(mean_imputed_data)
        ae_imputed_data = mean_imputed_data.copy()
        reconstructed_data_df = pd.DataFrame(reconstructed_data, columns=df.columns)
        ae_imputed_data[df.isnull()] = reconstructed_data_df[df.isnull()]

        rmse = sqrt(mean_squared_error(complete_values, ae_imputed_data))
        r2 = r2_score(complete_values, ae_imputed_data)

        sensors_final.append(sensor)

        perc = float(filename.split('-')[1])
        burst = int(filename.split('-')[2].split('.csv')[0])

        results.append({
            'perc': perc,
            'burst': burst,
            'sensor': str(sensors_final),
            'rmse': rmse,
            'r2': r2
        })
    
    result = pd.DataFrame(results)
    result.to_csv(f'results/multivariate/MICE-ae{index}-b2257.csv', sep=';', index=False)
    # print(result)

if __name__ == '__main__':
    results = []
    index = 0
    for _, _, files in os.walk('data'):
        for file in files:
            main(file, index)
            index += 1