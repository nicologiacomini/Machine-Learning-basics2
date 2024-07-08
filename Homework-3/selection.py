import matplotlib.pyplot as plt
import numpy as np

with open("performance_results-initial.csv", "r") as f:
    lines = f.readlines()
    r2_list = []
    rmse_list = []
    count_list = []
    feature_list_global = []
    model_names = []
    ref_model = None
    r2_list_model = []
    rmse_list_model = []
    feature_list_model = []
    count_model = []
    
    for line in lines[1:]:
        model_name, features, r2, rmse = line.split(";")
        r2 = float(r2)
        rmse = float(rmse.replace("\n", ""))
        list_string = features.split('[')[1].split(']')[0]
        feature_list = list_string.replace("'", "").split(', ')
        count = len(feature_list)

        if model_name == ref_model:
            r2_list_model.append(float(r2))
            rmse_list_model.append(float(rmse))
            feature_list_model.append(feature_list)
            count_model.append(count)
        else:
            model_names.append(model_name)
            if r2_list_model != []:
                r2_list.append(r2_list_model)
                rmse_list.append(rmse_list_model)
                feature_list_global.append(feature_list_model)
                count_list.append(count_model)
                r2_list_model = []
                rmse_list_model = []
                feature_list_model = []
                count_model = []
            r2_list_model.append(float(r2))
            rmse_list_model.append(float(rmse))
            feature_list_model.append(feature_list)
            count_model.append(count)
            ref_model = model_name
    r2_list.append(r2_list_model)
    rmse_list.append(rmse_list_model)
    feature_list_global.append(feature_list_model)
    count_list.append(count_model)


print(r2_list)
print(rmse_list)
# print(feature_list_global)
print(count_list)
print(model_names)

for i in range(len(model_names)):
    x = np.arange(1, 12, 1)
    plt.figure(figsize=(10, 8))
    plt.xticks(x)
    plt.plot(x, r2_list[i], '-o', label='R2')
    plt.plot(x, rmse_list[i], '-o', label='RMSE')
    plt.axvline(x=count_list[i][r2_list[i].index(max(r2_list[i]))], color='r', label='Best R2 and RMSE score', linestyle='--')
    plt.xlabel('Number of features')
    plt.ylabel('Score')
    plt.legend()
    plt.grid()
    plt.title(f'Model: {model_names[i]}')
    plt.savefig(f"r2-rmse/plot-{model_names[i]}.png")
    plt.close()
    print(f"Model {model_names[i]}")
    print(f"Best R2 score: {max(r2_list[i])} with {count_list[i][r2_list[i].index(max(r2_list[i]))]} features")
    print(f"Best RMSE score: {min(rmse_list[i])} with {count_list[i][rmse_list[i].index(min(rmse_list[i]))]} features")

print("\n")
list_r2_means = []
list_rmse_means = []
for j in range(7,11):
    sum_r2 = 0
    sum_rmse = 0
    for i in range(len(model_names)):
        if r2_list[i][j] > 0:
            sum_r2 += r2_list[i][j]
            sum_rmse += rmse_list[i][j]
    mean_r2 = sum_r2 / 4
    mean_rmse = sum_rmse / 4
    print(f"Mean R2 score for {j+1} features: {mean_r2}")
    print(f"Mean RMSE score for {j+1} features: {mean_rmse}")
    list_r2_means.append(mean_r2)
    list_rmse_means.append(mean_rmse)

print(f"The best result for r2 is {max(list_r2_means)} with {list_r2_means.index(max(list_r2_means))+8} features")
print(f"The best result for rmse is {min(list_rmse_means)} with {list_rmse_means.index(min(list_rmse_means))+8} features")
