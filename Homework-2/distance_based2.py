from get_data_info import GetDataInfo as gdi
from build_map import BuildMap as bm
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd

def assign_color(value):
    if value < 0.1:
        color = "yellow"
    elif 0.1 <= value < 0.2:
        color = "gold"
    elif 0.2 <= value < 0.3:
        color = "orange"
    elif 0.3 <= value < 0.4:
        color = "red"
    elif 0.4 <= value <= 0.5:
        color = "black"
    else:
        color = "black"
    return color

def normalize(matrix):
        # Find the minimum and maximum values in the matrix
        min_val = np.min(matrix)
        max_val = np.max(matrix)
    
        # Normalize the matrix
        normalized_matrix = (matrix - min_val) / (max_val - min_val)
        
        return normalized_matrix

datset = gdi.get_data_from_csv()

df = pd.read_csv("data/Node-Location.csv", delimiter=';', skiprows=0, low_memory=False)
points = {}
colors = []
for i in range(len(df)):
    lon = df['Lon'][i]
    lat = df['Lat'][i]
    name = df['Name'][i].title()
    points[name] = [lat, lon]

distance_matrix = gdi.get_distance_matrix()

sd = np.std(distance_matrix)

new_matrix = np.zeros((len(distance_matrix), len(distance_matrix)))
dim = new_matrix.shape
T = 5

connected_pair = []
colors_list = []
for i in range(dim[0]):
    for j in range(dim[1]):
        val = math.exp(-distance_matrix[i, j]**2 / 2*sd**2)
        if val < T:
            new_matrix[i, j] = val
            colors.append(assign_color(val))
            connected_pair.append((i, j))
        else:
            new_matrix[i, j] = 0

new_matrix = normalize(new_matrix)
points_name_list = list(points.keys())
points_coord_list = list(points.values())
connections = []
connections_list = []
for i in connected_pair:
    connections_list.append([points_coord_list[i[0]], points_coord_list[i[1]]])

connections.append(connections_list)
connections.append(colors)

bm.build_map(points, connections)