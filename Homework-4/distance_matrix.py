from math import radians, cos, sin, asin, sqrt
from sklearn.experimental import enable_iterative_imputer
import numpy as np
import pandas as pd

def haversine(lon1, lat1, lon2, lat2):
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

def ceate_distance_matrix(data):
    with open('Node-Location.csv', 'r') as file:
        lines = file.readlines()
        names = []
        lines = lines[1:]
        distance_matrix = [ [0]*len(lines) for _ in range(len(lines))]
        i= 0
        for line in lines:
            name, _, lat1, lon1 = line.split(';')
            names.append(name)
            j = 0
            for line2 in lines:
                name, _, lat2, lon2 = line2.split(';')
                val = haversine(float(lon1), float(lat1), float(lon2), float(lat2))
                distance_matrix[i][j] = round(val, 2)
                j += 1
            i += 1
    distance_matrix = np.array(distance_matrix)
    dataset = pd.DataFrame(distance_matrix, columns=names)
    dataset.insert(0, 'node', names)
    dataset.to_csv('distance_matrix.csv', sep=';', index=False)

def get_shortest_path():
    with open('distance_matrix.csv', 'r') as file:
        lines = file.readlines()
        lines = lines[1:]
        path = []
        for i in range(len(lines)):
            l = lines[i].split(';')
            l = l[1:]
            min = 1000
            for j in range(i+1, len(lines)):
                if float(l[j]) < min:
                    min = float(l[j])
            path.append(min)
    print(path)
                
                

if __name__ == '__main__':
    ceate_distance_matrix('Node-Location.csv')
    get_shortest_path()