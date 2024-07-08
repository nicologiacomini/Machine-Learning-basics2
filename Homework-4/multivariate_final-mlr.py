import pandas as pd
import os
from math import radians, cos, sin, asin, sqrt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import mean_squared_error, r2_score

def mice_imputation(data):
    df_copy = data.copy()
    missing_mask = df_copy.isna()

    imputer = IterativeImputer(max_iter=100, random_state=0)
    imputed_values = imputer.fit_transform(df_copy)
    new_df = pd.DataFrame(imputed_values, columns=df_copy.columns)

    df_copy[missing_mask] = new_df[missing_mask]
    return df_copy


def main(filename, index):
    missing_data_df = pd.read_csv(f'data/{filename}', sep=';').drop(columns=['date'])
    complete_data_df = pd.read_csv('data_matrix.csv', sep=';').drop(columns=['date'])

    # imputed_data = mice_imputation(missing_data_df)

    sensors_list = ["gracia", "eixample", "pr", "hebron", "ciutadella", "badalona", "montcada", "prat"]
    results = []
    df = missing_data_df[sensors_list[0]]
    complete_values = complete_data_df[sensors_list[0]]
    imputed_data = complete_data_df[sensors_list[0]]
    sensors_final = [sensors_list[0]]
    for sensor in sensors_list[1:]:
        # add to df missing_data_df[sensor]
        df = pd.concat([df, missing_data_df[sensor]], axis=1)
        imputed_data = mice_imputation(df)

        complete_values = pd.concat([complete_values, complete_data_df[sensor]], axis=1)

        rmse = sqrt(mean_squared_error(complete_values, imputed_data))
        r2 = r2_score(complete_values, imputed_data)

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
    result.to_csv(f'results/multivariate/MICE-mlr-100-{index}.csv', sep=';', index=False)
    # print(result)

if __name__ == '__main__':
    results = []
    index = 0
    for _, _, files in os.walk('data'):
        for file in files:
            main(file, index)
            index += 1
