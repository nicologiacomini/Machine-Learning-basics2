import csv
import numpy as np
import matplotlib.pyplot as plt 


def calculate_rmse(X, Y):
    difference = X - Y
    squared_diff = difference**2
    mean_square_diff = np.mean(squared_diff)
    rmse = np.sqrt(mean_square_diff)
    print(f"RMSE = {rmse}")
    return rmse

# read num_connections.csv
num_connections = []
thetas = []
thresholds = []
ratios = []
isFirst = True
with open('num_connections.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        if isFirst:
            isFirst = False
            continue
        num_connections.append(float(row[3]))
        thetas.append(float(row[1]))
        ratios.append(float(row[2]))
        thresholds.append(float(row[0]))

# normalize datasets
num_connections = np.array(num_connections).astype(float)
num_connections = (num_connections - num_connections.mean()) / num_connections.std()

thetas = np.array(thetas).astype(float)
thetas = (thetas - thetas.mean()) / thetas.std()

thresholds = np.array(thresholds).astype(float)
thresholds = (thresholds - thresholds.mean()) / thresholds.std()

ratios = np.array(thresholds**2 / 2*thetas**2).astype(float)

# plot the correlation matrix
def analysis(first_data, second_data):
    calculate_rmse(np.array(first_data), np.array(second_data))

    plt.figure(figsize=(10, 8))
    plt.scatter(first_data, second_data, color='blue')
    plt.xlabel('Number of connections')
    plt.ylabel('Thresholds')
    M = max(max(first_data), max(second_data))
    m = min(min(first_data), min(second_data))
    plt.plot([m, M], [m, M], ls="--",c='red')
    plt.grid(True)  # Add grid
    plt.title('Relationship between number of connections and thresholds')
    plt.show()

analysis(num_connections, thetas)