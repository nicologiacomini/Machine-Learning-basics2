import pandas as pd
import matplotlib.pyplot as plt

name = "SVR"

df = pd.read_csv(f"final_results_{name}-r2-rmse.csv", sep="; ")
path = f"analysis/{name}/"
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 10))

# get the row with the best r2
best_r2 = df.loc[df['r2'].idxmax()]
print(best_r2.to_list())

# List of components model; shuffled; n_neighbors; weights; algorithm; leaf_size; r2; rmse
# components = ['shuffled', 'kernel', 'n_restart_optimizer', 'alpha', 'normalize_y'] # GP
# components = ['shuffled', 'n_neighbors', 'weights', 'algorithm', 'leaf_size'] # KNN
components = ['shuffled', 'kernel', 'alpha', 'gamma'] # KR
# components = ['shuffled', 'hidden_layers', 'units', 'learning_rate', 'epochs', 'dropout_rate'] # FFNN
# components = ['shuffled', 'max_depth', 'n_estimators', 'min_samples_split', 'min_samples_leaf', 'bootstrap'] # RF
# components = ['shuffled', 'c', 'gamma', 'epsilon'] # SVR


# Plot R2 and RMSE for each component
for idx, ax in enumerate(axes.flatten()):
    if idx >= len(components):
        break
    component = components[idx]
    ax.scatter(df[component], df['r2'], label='R²', color='b', marker='o')
    ax.scatter(df[component], df['rmse'], label='RMSE', color='r', marker='x')
    ax.scatter(best_r2[component], best_r2['r2'], color='lime', marker='o', label='Best R²')
    ax.scatter(best_r2[component], best_r2['rmse'], color='lime', marker='x', label='Best RMSE')
    ax.set_xlabel(component)
    ax.set_ylabel('Value')
    ax.legend()

# Adjust layout
plt.tight_layout()
plt.savefig(f"{path}r2_rmse.png")