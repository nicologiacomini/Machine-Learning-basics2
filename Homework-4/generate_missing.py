import pandas as pd
import missing_generator as mg
import json
import numpy as np 
import csv


perc_cases = [0.1, 0.2, 0.3, 0.4, 0.5]
length_cases = [5, 10, 20]

for perc in perc_cases:
    for length in length_cases:
        # Load data
        data = pd.read_csv('data_matrix.csv', sep=';')

        data_list = []
        missing_dict = {}

        data_list.append(data['date'])
        pandas_list = pd.DataFrame({'date': data['date']})

        name_list = ['date']

        # Generate missing values
        sensor_values = data.drop('date', axis=1)
        for column in sensor_values.columns:
            indexes_missing, incomplete_time_series = mg.produce_missings(data[column], perc, length)
            missing_dict[column] = indexes_missing.tolist()
            
            name_list.append(column)
            data_list.append(incomplete_time_series)
            data_to_load = np.array(data_list).T
            df=pd.DataFrame(data_to_load,columns=name_list)

        with open(f'missings/missing_data_dict-{perc}-{length}.json', 'w') as file:
            json.dump(missing_dict, file)

        df.to_csv(f'data/incomplete_data_matrix-{perc}-{length}.csv', sep=';', index=False)