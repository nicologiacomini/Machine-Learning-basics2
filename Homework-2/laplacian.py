from get_data_info import GetDataInfo as gdi
from build_map import BuildMap as bm
from get_matrices import GetMatrices as gm
import numpy as np
import pandas as pd
import cvxpy as cp
from sklearn.preprocessing import StandardScaler


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
lambda_list = [0.1, 0.2, 0.3, 0.4, 0.5, 1, 2, 3, 4, 5]

# for k in range(len(threshold_list)):
k = 0
h = k
t = k

l = 0.08
n = 8
th = 0 # value between 0 and 1

# W, X, L = gm.get_matrices(5, 0.5)

X = gdi.get_data_from_csv()

# standardize the data
scaler = StandardScaler().fit(X)
X = scaler.transform(X)

n, m = X.shape

# print the type of matrix X
change = X.tolist()
print(change[7][7])
print(type(change))

beta_list = [1, 10, 11, 12, 13, 15, 22, 30, 34, 40, 55, 70, 100, 140, 250, 500, 1000]
beta_list = [1, 1, 10, 100, 1000]
alpha_list = [1, 10]
# beta_list of integer number from 50 to 100 with step 5

for index in range(len(beta_list)):
    for index_ in range(len(alpha_list)):
        # print("\nBETA: ", beta_list[index])
        alpha = alpha_list[index_]
        beta = beta_list[index]

        Y = cp.Parameter((n, m), value=X.T.tolist())
        # Y = cp.Variable((n, m))
        L = cp.Variable((n, n), symmetric=True)
        # L = cp.Variable((n, n), symmetric=True)
        # Y = X

        constraints = [
            cp.trace(L) == n,
            L == L.T,  # L must be symmetric
            # L <= 0,  # Lij = Lji ≤ 0, i != j
            cp.sum(L, axis=1) == 0  # L·1 = 0
        ]

        for i in range(n):
            for j in range(n):
                if i != j:  # Exclude diagonal elements
                    constraints.append(L[i, j] <= 0)

        # objective = cp.Minimize(cp.norm(X - Y, 'fro') + alpha * cp.trace(Y.T @ L @ Y) + beta * cp.norm(L, 'fro'))
        objective = cp.Minimize(cp.norm(X - Y.value, 'fro')**2 + alpha * cp.trace(cp.quad_form(Y.value, L)) + beta * cp.norm(L, 'fro')**2)
        # objective = cp.Minimize(alpha * cp.trace(cp.quad_form(Y.value, L)) + beta * cp.norm(L, 'fro'))
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.SCS, verbose=False)

        # round the solution to 3 decimal places
        solution = np.round(L.value, 3)
        print("Optimal L:\n", solution)
        if L.value is not None:
            print("trace: ", np.trace(L.value))

        L_optimized = L.value
        print("dim = ", L_optimized.shape)

        Y = cp.Variable((n, m))
        print("Y: ", Y.shape)
        L_fixed = cp.Parameter((n, n), value=L_optimized, symmetric=True)

        new_objective = cp.Minimize(cp.norm(X - Y, 'fro')**2 + alpha * cp.trace(cp.quad_form(Y.T, L_optimized)) + beta * cp.norm(L_optimized, 'fro')**2)
        new_problem = cp.Problem(new_objective)
        new_problem.solve(solver=cp.ECOS_BB, verbose=True)

        print("Optimal Y:\n", Y.value)


        th = 0.1
        count = 0
        for i in range(n):
            for j in range(n):
                val = abs(L_optimized[i, j])
                if val < th:
                    L_optimized[i, j] = 0
                else:
                    L_optimized[i, j] = val
        
        for i in range(n):
            for j in range(i, n):
                if L_optimized[i, j] > th and i != j:
                    count += 1
        
        print("Case of alpha: ", alpha, " and beta: ", beta)
        print("Number of connections: ", count)
        
        output = "laplacian/alpha_" + str(alpha) + "_beta_" + str(beta) + ".txt"
        with open(output, 'w') as f:
            print("Optimal L:\n", L_optimized, file=f)
            print("trace: ", np.trace(L_optimized), file=f)
            

        dim = L_optimized.shape
        name = "laplacian/alpha_" + str(alpha) + "_beta_" + str(beta)

        connected_pair = []
        colors_list = []
        for i in range(dim[0]):
            for j in range(i, dim[1]):
                val = L_optimized[i, j]
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