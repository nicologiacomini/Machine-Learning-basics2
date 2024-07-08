import pandas as pd
import os
from math import radians, cos, sin, asin, sqrt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import KNNImputer

def impute_with_model(df, model):
    df_imputed = df.copy()
    for column in df.columns:
        # Set placeholder back to missing
        missing_idx = df[column].isna()
        if missing_idx.sum() == 0:
            continue
        
        # Prepare data for regression/KNN
        train_data = df_imputed[~missing_idx]
        test_data = df_imputed[missing_idx]
        
        X_train = train_data.drop(columns=[column])
        y_train = train_data[column]
        X_test = test_data.drop(columns=[column])
        
        # Fit model and predict missing values
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Replace missing values with predictions
        df_imputed.loc[missing_idx, column] = y_pred
    
    return df_imputed


def mice_imputation(data, n):
    df_copy = data.copy()
    missing_mask = df_copy.isna()
    mean_imputed_data = data.fillna(data.mean())

    knn_imputer = KNNImputer(n_neighbors=n)
    knn_imputed_data = impute_with_model(mean_imputed_data, knn_imputer)
    new_df = pd.DataFrame(knn_imputed_data, columns=df_copy.columns)

    df_copy[missing_mask] = new_df[missing_mask]
    return df_copy


def main(filename, index):
    # Step 0: load the datasets
    missing_data_df = pd.read_csv(f'data/{filename}', sep=';').drop(columns=['date'])
    complete_data_df = pd.read_csv('data_matrix.csv', sep=';').drop(columns=['date'])

    # imputed_data = mice_imputation(missing_data_df)

    # Step 2: Calculate the RMSE and R2
    results = []
    k_list = [1, 3, 5, 7]

    for k in k_list:
        # add to df missing_data_df[sensor]
        imputed_data = mice_imputation(missing_data_df, k)

        rmse = sqrt(mean_squared_error(complete_data_df, imputed_data))
        r2 = r2_score(complete_data_df, imputed_data)

        perc = float(filename.split('-')[1])
        burst = int(filename.split('-')[2].split('.csv')[0])

        results.append({
            'perc': perc,
            'burst': burst,
            'neighborhood': k,
            'rmse': rmse,
            'r2': r2
        })
    
    result = pd.DataFrame(results)
    result.to_csv(f'results/multivariate/MICE-knn{index}.csv', sep=';', index=False)
    # print(result)

if __name__ == '__main__':
    results = []
    index = 0
    for _, _, files in os.walk('data'):
        for file in files:
            main(file, index)
            index += 1