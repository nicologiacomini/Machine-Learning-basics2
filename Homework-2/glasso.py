from get_data_info import GetDataInfo as gdi
from build_map import BuildMap as bm
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import cvxpy as cp
from sklearn.preprocessing import StandardScaler
import time

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

lambda_list = [0.01, 0.02, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 2, 3, 4, 5]

for k in range(len(lambda_list)):
    # k = 0
    # h = k
    # t = k
    # l = k

    l = lambda_list[k]
    n = 8
    th = 0 # value between 0 and 1

    # weight_matrix = gdi.get_weight_matrix(threshold_list[h], theta_list[l])
    x_matrix = gdi.get_data_from_csv()
    x_matrix = pd.DataFrame(x_matrix).to_numpy().astype(float)

    # normalize x_matrix
    x_matrix = (x_matrix - x_matrix.mean(axis=0)) / x_matrix.std(axis=0)
    sample_covariance = np.cov(x_matrix.T, rowvar=False)

    # standardize the data
    scaler = StandardScaler().fit(x_matrix)
    x_matrix = scaler.transform(x_matrix)
    sample_covariance = np.cov(x_matrix.T, rowvar=False)

    Theta = cp.Variable((n, n), symmetric=True)
    objective = cp.Minimize(cp.trace(sample_covariance @ Theta) - cp.log_det(Theta) + l * cp.sum(cp.abs(Theta - np.eye(n))))
    constraints = [Theta >= 0]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS, verbose=False)

    final_theta = np.around(Theta.value, decimals=2)

    for i in range(n):
        for j in range(i,n):
            if final_theta[i, j] < th and i != j:
                final_theta[i, j] = 0
            # if final_theta[i, j] > 1 and i != j:
                # print("Error: ", final_theta[i, j])

    count = 0
    for i in range(n):
        for j in range(i, n):
            if final_theta[i, j] > th and i != j:
                count += 1

    # print("Optimal Theta:\n", final_theta)
    print(f"Case of lambda: {l} and threshold: {th}")
    print(f"Number of connections: {count}")


    dim = final_theta.shape
    name = "glasso/lambda_" + str(l) + "_threshold_" + str(th)

    connected_pair = []
    colors_list = []
    for i in range(dim[0]):
        for j in range(i, dim[1]):
            val = final_theta[i, j]
            if val != 0 and i != j:
                colors_list.append(assign_color(val))
                connected_pair.append((i, j))


    points_name_list = list(points.keys())
    points_coord_list = list(points.values())
    connections = []
    connections_list = []
    for i in connected_pair:
        connections_list.append([points_coord_list[i[0]], points_coord_list[i[1]]])

    connections.append(connections_list)
    connections.append(colors_list)

    bm.build_map(points, connections, name)

    '''
    COMMENT:
    The algorithm loses significance if we use a value to small, in this case: 0.3. Because the values different from the diagonal are too high, and overcomes the 1.
    In the same way also if we use a value too big, in this case: 0.7, because all the values different from the diagonal are too small, and the graph is not connected.
    The bests are 0.04 and 0.05, with a threshold of 0.1.
    '''