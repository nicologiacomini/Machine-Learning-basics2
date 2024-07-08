from get_data_info import GetDataInfo as gdi
from build_map import BuildMap as bm
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd

def assign_color(value):
    if value < 0.2:
        color = "yellow"
    elif 0.2 <= value < 0.4:
        color = "gold"
    elif 0.4 <= value < 0.6:
        color = "orange"
    elif 0.6 <= value < 0.8:
        color = "red"
    elif 0.8 <= value <= 1.0:
        color = "black"
    else:
        color = "black"
    return color

df = pd.read_csv("data/Node-Location.csv", delimiter=';', skiprows=0, low_memory=False)
points = {}
colors = []
for i in range(len(df)):
    lon = df['Lon'][i]
    lat = df['Lat'][i]
    name = df['Name'][i].title()
    points[name] = [lat, lon]

threshold_list = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30]
theta_list = [0.1, 0.2, 0.3, 0.4, 0.5, 1, 2, 3, 4, 5]

threshold_list = [5]
theta_list = [0.1]

for k in range(len(threshold_list)):
    for p in range(len(theta_list)):
        h = p
        l = k
        weight_matrix = gdi.get_weight_matrix(threshold_list[h], theta_list[l])
        dim = weight_matrix.shape
        name = "distance-based-output/threshold_" + str(threshold_list[h]) + "_theta_" + str(theta_list[l])

        connected_pair = []
        colors_list = []
        count = 0
        for i in range(dim[0]):
            for j in range(i, dim[1]):
                val = weight_matrix[i, j]
                if val != 0 and i != j:
                    colors_list.append(assign_color(val))
                    connected_pair.append((i, j))
                    count += 1

        print(f"Case of threshold: {threshold_list[h]} and theta: {theta_list[l]}")
        print(f"Number of connections: {count}")

        # print(np.min(weight_matrix), np.max(weight_matrix))

        # points_name_list = list(points.keys())
        # points_coord_list = list(points.values())
        # connections = []
        # connections_list = []
        # for i in connected_pair:
        #     connections_list.append([points_coord_list[i[0]], points_coord_list[i[1]]])

        # connections.append(connections_list)
        # connections.append(colors_list)

        # bm.build_map(points, connections, name)