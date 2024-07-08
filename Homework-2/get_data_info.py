from math import radians, cos, sin, asin, sqrt
import csv
import math
import numpy as np
import pandas as pd

class GetDataInfo:

    @staticmethod
    def __haversine(lon1, lat1, lon2, lat2):
        """
        Calculate the great circle distance in kilometers between two points
        on the earth (specified in decimal degrees)
        """
        # convert decimal degrees to radians
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

        # haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        r = 6371 # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
        return c * r
    
    @staticmethod
    def __normalize(matrix):
        # Find the minimum and maximum values in the matrix
        min_val = np.min(matrix)
        max_val = np.max(matrix)
    
        # Normalize the matrix
        normalized_matrix = (matrix - min_val) / (max_val - min_val)
        
        return normalized_matrix

    @staticmethod
    def __read_csv(file_path):
        with open(file_path, 'r') as f:
            reader = csv.reader(f, delimiter=';')
            data = list(reader)
            data.pop(0)
            return data

    @staticmethod
    def get_weight_matrix(threshold=10, theta=10):
        data = GetDataInfo.__read_csv('data/Node-Location.csv')

        # build the distance matrix
        weight_matrix = np.zeros((len(data), len(data)))
        for i in range(len(data)):
            for j in range(len(data)):
                dist = GetDataInfo.__haversine(float(data[i][3]), float(data[i][2]), float(data[j][3]), float(data[j][2]))
                if dist < threshold:
                    weight_matrix[i, j] = math.exp(-dist**2 / 2*theta**2)
                    if i > j:
                        print(f"Distance between {data[i][0]} and {data[j][0]}: {dist}")
                else:
                    weight_matrix[i, j] = 0

        # distances = GetDataInfo.__normalize(distances)
        return weight_matrix 
    
    def get_relative_positions():
        data = GetDataInfo.__read_csv('data/Node-Location.csv')
        positions = []
        for i in range(len(data)):
            positions.append([data[i][2], data[i][3]])
        return positions
    
    @staticmethod
    def get_data_from_csv():
        data_gracia = []
        data_pr = []
        data_eixample = []
        data_prat = []
        data_montcada = []
        data_ciutadella = []
        data_hebron = []
        data_badalona = []

        dataset = [[], [], [], [], [], [], [], []]
        isFirst = True
        with open('data/data_matrix.csv', 'r') as f:
            reader = csv.reader(f, delimiter=';')
            for row in reader:
                if isFirst:
                    isFirst = False
                    continue
                dataset[0].append(row[1])
                dataset[1].append(row[2])
                dataset[2].append(row[3])
                dataset[3].append(row[4])
                dataset[4].append(row[5])
                dataset[5].append(row[6])
                dataset[6].append(row[7])
                dataset[7].append(row[8])

                data_gracia.append(row[1])
                data_pr.append(row[2])
                data_eixample.append(row[3])
                data_prat.append(row[4])
                data_montcada.append(row[5])
                data_ciutadella.append(row[6])
                data_hebron.append(row[7])
                data_badalona.append(row[8])

        df = pd.DataFrame(dataset)
        df.reset_index(drop=True)

        # return data_gracia, data_pr, data_eixample, data_prat, data_montcada, data_ciutadella, data_hebron, data_badalona
        return df
    