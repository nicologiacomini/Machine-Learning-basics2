import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

prefixed = [filename for filename in os.listdir('final-results/ae') if (filename.startswith("MICE-ae") and filename.endswith(".csv"))]
list_index = [0, 1, 2, 3, 4]
 
sensor_list = ['gracia', 'pr', 'eixample', 'prat', 'montcada', 'ciutadella', 'hebron', 'badalona', 'montcada', 'missing-burst']

global_r2_list = []
global_rmse_list = []
global_perc_burst_list = []

for file in prefixed:
    file = f'final-results/ae/{file}'
    df = pd.read_csv(file, sep=';')
    global_perc_burst_list.append(str(df.iat[0, 0])+" "+str(df.iat[0, 1]))

for s in sensor_list[:-1]:
    r2_list = []
    rmse_list = []
    perc_burst_list = []
    for file in prefixed:
        file = f'final-results/ae/{file}'
        df = pd.read_csv(file)
        for i in range(len(df)):
            if df.iat[i, 0] == s:
                r2_list.append(df.iat[i, 4])
                rmse_list.append(df.iat[i, 3])
                break

    global_r2_list.append(r2_list)
    global_rmse_list.append(rmse_list)

r2_list = []
rmse_list = []

for file in prefixed:
    file = f'final-results/ae/{file}'
    df = pd.read_csv(file, sep=';')
    for i in range(len(df)):
        r2_list.append(df.iat[i, 4])
        rmse_list.append(df.iat[i, 3])
        break


global_r2_list.append(global_perc_burst_list)
global_rmse_list.append(global_perc_burst_list)

r2_list = np.array(global_r2_list)
rmse_list = np.array(global_rmse_list)

r2_list = np.array(r2_list)
rmse_list = np.array(rmse_list)

header = ['perc', 'burst', 'sensor', 'rmse', 'r2']

r2_list = pd.DataFrame(r2_list.T, columns=header)
rmse_list = pd.DataFrame(rmse_list.T, columns=header)

new_r2_list = r2_list.drop(columns=['missing-burst'])
# make all the values floats
new_r2_list = new_r2_list.astype(float)
# remove all the rows where there is a negative value
new_r2_list = new_r2_list[(new_r2_list.T >= 0).all()]
print(new_r2_list)

average_r2 = new_r2_list.mean()
print(average_r2)

fig = plt.figure()
plt.bar(sensor_list[:-1], average_r2)
plt.xlabel('Sensor')
plt.ylabel('Average R2')
plt.title('Average R2 for each sensor')
plt.show()


new_rmse_list = rmse_list.drop(columns=['missing-burst'])
# make all the values floats
new_rmse_list = new_rmse_list.astype(float)
# remove all the rows where there is a negative value
new_rmse_list = new_rmse_list[(new_rmse_list.T >= 0).all()]
print(new_rmse_list)

average_rmse = new_rmse_list.mean()
print(average_rmse)

fig = plt.figure()
plt.bar(sensor_list[:-1], average_rmse)
plt.xlabel('Sensor')
plt.ylabel('Average RMSE')
plt.title('Average RMSE for each sensor')
plt.show()